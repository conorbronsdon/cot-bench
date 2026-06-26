"""Multi-turn agent simulation runner.

Orchestrates conversations between a user simulator, the agent under test,
and a tool simulator. Tracks tokens, latency, and full conversation history
for downstream scoring.

The agent loop uses LangChain native tool calling: scenario tool definitions
are converted to OpenAI-style JSON Schema function definitions and bound to the
model via ``bind_tools``. Within a single user turn the agent iterates
agent -> tool -> agent until it produces a user-facing message (no tool calls)
or hits the inner tool-round cap. Loop structure and tool-schema conversion are
adapted from Galileo's agent-leaderboard (Apache 2.0), acknowledged in the README.
"""

import copy
import json
import logging
import re
import time
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from eval.config import DEFAULT_SIMULATION, DOMAIN_CONFIGS, Domain, SimulationConfig
from eval.providers.registry import ModelSpec, create_model
from eval.scoring.state_check import score_state_changes
from eval.simulation.dual_control import DualControl, action_fires
from eval.simulation.probes import RecoveryProbe
from eval.simulation.profiles import DEFAULT_SIM_PROFILE, profile_instructions
from eval.simulation.tool_transitions import get_transition

logger = logging.getLogger(__name__)

CONVERSATION_COMPLETE = "[CONVERSATION_COMPLETE]"

# Max agent<->tool rounds within a single user turn before we force a hand-back
# to the user. Prevents an agent that keeps emitting tool calls from looping
# forever inside one turn. max_turns still bounds the number of USER turns.
MAX_TOOL_ROUNDS_PER_TURN = 5

# JSON-in-text tool-call pattern. Native tool calling is the primary path; this
# is kept ONLY as a clearly-labeled fallback for providers that return no native
# tool_calls but emit a tool call embedded in content (matching the old repo
# protocol). When it fires we log a warning so its frequency is measurable.
# Handles up to ~3 levels of brace nesting.
_TOOL_CALL_RE = re.compile(
    r'\{\s*"tool_call"\s*:\s*(\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\})\s*\}'
)

# JSON Schema types we accept from scenario parameter definitions.
_JSON_SCHEMA_TYPES = {"string", "integer", "number", "boolean", "array", "object"}


@dataclass
class ToolCall:
    """A single tool call made by the agent."""

    turn: int
    tool_name: str
    arguments: dict
    result: str
    tool_call_id: str = ""


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""

    turn_number: int
    role: str  # "user", "agent", "tool"
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    latency_ms: float = 0.0
    token_count: int = 0
    tool_call_id: str = ""  # set on "tool" turns, matches the issuing ToolCall


@dataclass
class SimulationResult:
    """Complete result from running one scenario."""

    scenario_id: str
    domain: str
    model_name: str
    turns: list[ConversationTurn]
    total_turns: int
    total_latency_ms: float
    total_input_tokens: int
    total_output_tokens: int
    completed: bool  # Whether conversation reached natural completion
    error: str | None = None
    # --- User-sim completion decoupling (#32, part 1) ---
    # How the conversation ended, decoupled from whether the user sim was
    # satisfied. The user simulator no longer gets to DECLARE the goals met; it
    # only signals it is done talking. Whether the goals were actually met is the
    # deterministic state check, recorded separately so a miscalibrated sim that
    # ends early becomes visible in the artifact rather than silently passing.
    #   - "user_sim"  : the user simulator emitted CONVERSATION_COMPLETE.
    #   - "max_turns" : the outer turn budget was exhausted with no sim signal.
    #   - "error"     : the agent raised and the run aborted.
    ended_by: str = "max_turns"
    # Deterministic state-check pass fraction (n_passed / n_total) computed
    # against the mutated world AT THE MOMENT the conversation ended. None when
    # the scenario carries no ground_truth (state grading inapplicable) — those
    # scenarios still rely on the judges, and the sim signal is all we have.
    state_progress_at_end: float | None = None
    # True iff the user sim ended the conversation (ended_by == "user_sim") while
    # the deterministic state check was still below 1.0 — i.e. the sim declared
    # itself done before the goals were verifiably met. This is the premature-
    # ending signal that aggregate_results can turn into a premature-ending rate.
    premature_end: bool = False
    # Final mutable world state at end of run. None for legacy (stateless)
    # scenarios that carry no ground_truth.
    final_world: dict | None = None
    # Provider-reported model id from the agent's responses (first one seen).
    # The configured model_id pins what we ASKED for; this records what the
    # provider says it SERVED — they can differ on alias/routed providers
    # (OpenRouter does not pin the upstream provider or quantization).
    resolved_model: str | None = None
    # Simulator-side token usage, summed across all user-sim + tool-sim calls in
    # this conversation (issue #47). Separate from the agent token totals above so
    # the running cost guard can price the simulators at THEIR model id (which can
    # be overridden per run, issue #50) rather than the agent's. Zero when a
    # provider returns no usage_metadata.
    sim_input_tokens: int = 0
    sim_output_tokens: int = 0
    # Resolved sim model ids (requested form) for this run — recorded so the sim
    # tokens above can be priced and the sensitivity-test delta (#50) is
    # attributable. Filled from the runner's SimulationConfig.
    user_sim_model: str | None = None
    tool_sim_model: str | None = None
    # Behavioral user-sim profile active for this run (issue #59 part 1) —
    # "cooperative" (the unchanged default), "impatient", "technically-confused",
    # or "adversarial". Stamped per row/artifact so persona-stratified pass rates
    # are computable and non-cooperative rows can be excluded from the public
    # leaderboard aggregates. Filled from the runner's SimulationConfig.
    sim_profile: str = DEFAULT_SIM_PROFILE
    # Dual control (issue #58). True when the scenario declared a ``dual_control``
    # block — both the agent AND the (scripted) user acted on the shared world
    # this run. False for the single-control majority (the entire v1 corpus; demo
    # fixtures live in tests only). When True, ``user_actions_fired`` counts how
    # many declared user actions actually fired (their trigger was met before the
    # conversation ended), and ``coordination_ok`` is the deterministic verdict:
    # did the agent reach the correct end state given the user's concurrent
    # actions, WITHOUT mutating a user-owned path the user already handled (no
    # double-apply, correct handoff). Both feed a SEPARATE dual_control table —
    # they never move public efficacy (no dual-control rows exist in v1).
    dual_control: bool = False
    user_actions_fired: int = 0
    # Deterministic coordination verdict for a dual-control run. None when the
    # scenario was not dual-control OR no user action fired (no coordination ever
    # occurred to grade) OR the scenario carries no ground_truth. For a run where
    # at least one user action fired, True/False is the AND of (a) the scenario's
    # normal expected_state_changes passing — the agent reached the correct end
    # state given the user's concurrent actions — and (b) the attribution contract
    # holding: every path the agent itself mutated was agent-authorized (the agent
    # did not write a user-owned path the user already handled).
    # compute_dual_control_rates drops None rows, so non-firing rows are excluded
    # from the coordination rate.
    coordination_ok: bool | None = None
    # Count of user actions whose trigger was met but whose firing was SUPPRESSED
    # because no delivery turn remained (the trigger was first satisfied on the
    # loop's final outer turn, so the staged user message could never be delivered
    # and the agent would get no turn to act on it — the #74 fired-but-not-
    # delivered class). A suppressed action is treated as NOT fired: its delta is
    # not applied, it is not counted in ``user_actions_fired``, and it grades no
    # coordination verdict (coordination_ok stays None when nothing else fired).
    # Surfaced on rows/artifacts so the trigger-met-but-undeliverable case is
    # auditable, exactly like the recovery-probe ``probe_fired`` flag.
    user_actions_suppressed: int = 0
    # Recovery probe (issue #57). When the scenario carried a recovery_probe, this
    # is its kind (contradictory_reference / wrong_entity / incomplete_action_claim)
    # and ``recovered`` is the deterministic verdict: did the agent reach the
    # correct end state DESPITE the injected fault. Both None for the vast
    # majority of rows — only probe-carrying scenarios populate them — so a
    # per-model recovery_rate is computed over probe rows ONLY and the public
    # aggregates (which see no probe rows in v1) are untouched.
    recovery_probe_kind: str | None = None
    # Deterministic recovery verdict for a FIRED probe. None when no probe was
    # declared OR the probe never fired (the conversation ended before
    # ``probe.turn`` — early CONVERSATION_COMPLETE, max_turns below the probe
    # turn, or an agent error before injection) OR the scenario carries no
    # ground_truth to grade against. For a fired probe, True/False is the AND of
    # (a) the scenario's normal expected_state_changes passing — the task still
    # got done despite the fault — and (b) the probe's own recovery_assertions
    # passing (typically the "did NOT act on the bad entity" check). A
    # never-fired probe MUST be None, never False: grading it would score a
    # fault that was never injected (and could even credit a recovery the agent
    # never performed). compute_recovery_rates drops None rows, so non-fired
    # rows are excluded from recovery_rate.
    recovered: bool | None = None
    # True iff the probe's injected message was actually delivered as a user
    # turn (the conversation reached ``probe.turn``). Always False when no probe
    # was declared. Surfaced on rows/artifacts alongside ``recovered`` so the
    # recovery_rate denominator is auditable: a declared-but-never-fired probe
    # row carries probe_fired=False and recovered=None.
    probe_fired: bool = False
    # Count of tool-sim responses that could not be parsed into a
    # {response, state_delta} object (S3). Such a call feeds raw text back to the
    # agent but applies NO state_delta, so the final world is missing whatever
    # mutation that call should have made. We count them per run and surface the
    # count on the row + artifact (mirroring user_actions_suppressed / probe_fired
    # auditing) so a graded world known to be incomplete is never silently scored:
    # on a STATEFUL scenario (one with ground_truth) a non-zero count makes the
    # state grade non-gradable (see build_result_row's state_gradable). 0 for a
    # clean run and for every stateless (no ground_truth) scenario.
    tool_sim_parse_failures: int = 0
    # Coded-vs-LLM tool authority split (issue #87, phase 1b). Every stateful tool
    # call is served either by a DETERMINISTIC coded transition (the world mutation
    # is a pure function of args+world) or by the LLM tool simulator (fallback for
    # a tool with no registered transition). These two counters surface that split
    # per run so a published leaderboard can report the fraction of the graded
    # world that was deterministically mutated — the audit-S2 claim #87 exists to
    # make true. Both 0 for a stateless (no ground_truth) run, which never engages
    # either path.
    coded_transition_calls: int = 0
    llm_tool_sim_calls: int = 0
    # Phase-3 spine-trust guard (#87): how many of the LLM-fallback tool calls
    # actually MUTATED the graded world (applied a non-empty state_delta). A
    # read-only fallback does not count. >0 on a stateful scenario makes the state
    # grade non-gradable (see is_state_gradable) — the spine refuses to grade a
    # world an unseeded LLM partly authored. 0 for a stateless run and for any run
    # whose every tool was a deterministic coded transition (the corpus norm).
    llm_tool_sim_mutations: int = 0


@dataclass
class Scenario:
    """A test scenario to run."""

    id: str
    domain: Domain
    persona: dict
    user_goals: list[str]
    tools: list[dict]  # repo tool definitions (name/description/parameters list)
    category: str
    initial_message: str
    # v0.2 stateful world. ``ground_truth`` is the canonical initial world the
    # tool simulator answers from and mutates; ``expected_state_changes`` are the
    # deterministic post-conversation assertions. Both None for legacy scenarios.
    ground_truth: dict | None = None
    expected_state_changes: list | None = None
    # Atomic rubric criteria (issue #54): 3-6 instance-specific, checkable
    # criteria informing the JUDGE dimensions only (task_completion /
    # tool_selection). None for scenarios without criteria, whose judge prompts
    # and scoring stay byte-identical to the templates. Unlike ``holdout`` below,
    # criteria ARE hashed scenario content when present (they change scoring
    # semantics); see _scenario_to_canonical_dict, which includes them
    # conditionally so criteria-less scenarios keep their existing digests.
    rubric_criteria: list | None = None
    # Private-holdout flag (issue #31). True when the scenario was loaded from the
    # external holdout directory rather than the public corpus. It tags result
    # rows so aggregation can compute a public-vs-holdout gap, and it gates
    # holdout content out of the published per-scenario surfaces. It is NOT part
    # of the hashed scenario content (see _scenario_to_canonical_dict): the
    # holdout's tamper-evidence comes from its own corpus hash, not from a
    # per-scenario flag that would differ between the public and holdout loaders.
    holdout: bool = False
    # Dual control (issue #58): OPTIONAL. When present, declares ``user_tools``
    # (what the user side may touch) and scripted ``user_actions`` (trigger -> a
    # user tool call with a state delta), so the SIMULATED USER also acts on the
    # shared world. The runner fires each action deterministically, applies its
    # delta through the SAME apply_state_delta the agent's tools use (one world),
    # and records the user-mutated paths so attribution stays sharp. None for the
    # single-control majority — the entire v1 corpus (demo fixtures live in test
    # fixtures only; see docs/dual-control.md). Like ``rubric_criteria`` it IS
    # hashed scenario content when present (it changes what the run does), added
    # conditionally to the canonical dict so single-control scenarios keep their
    # existing digests.
    dual_control: DualControl | None = None
    # Recovery probe (issue #57): an OPTIONAL deterministic mid-conversation
    # perturbation. When present, the runner injects ``probe.injection`` verbatim
    # at ``probe.turn`` (replacing the user sim's generated message that turn) and
    # grades recovery at end via state checking. None for every scenario without a
    # probe — which is the entire v1 corpus (demo probes live in test fixtures
    # only; see docs/recovery-probes.md). Like ``rubric_criteria`` it IS hashed
    # scenario content when present (it changes what the run does), and is added
    # conditionally to the canonical dict so probe-less scenarios keep their
    # existing digests.
    recovery_probe: RecoveryProbe | None = None


def apply_state_delta(world: dict, delta: dict) -> None:
    """Apply a tool-sim ``state_delta`` to ``world`` in place (deterministic).

    ``delta`` maps a dotted path to a new value, e.g.::

        {"accounts.BUS-CHK-001.balance": 10920.55}

    Dotted segments walk nested dicts; intermediate dicts are created on demand
    so a path can set a previously-absent key. Two conventions:

    - A plain value at ``path`` REPLACES whatever is at that path.
    - A value of the form ``{"__append__": <item>}`` APPENDS ``<item>`` to the
      list at ``path`` (creating an empty list first if the key is absent). This
      is the only way to grow a list — it keeps list mutation explicit and
      deterministic rather than guessing append-vs-replace from the value type.

    Invalid paths (e.g. trying to descend into a non-dict) are logged and
    skipped — a malformed delta must never crash a run.
    """
    if not isinstance(delta, dict):
        logger.warning("Ignoring non-dict state_delta: %r", delta)
        return

    for dotted, value in delta.items():
        parts = str(dotted).split(".")
        node = world
        ok = True
        # Walk to the parent of the final segment, creating dicts as needed.
        for key in parts[:-1]:
            nxt = node.get(key)
            if nxt is None:
                nxt = {}
                node[key] = nxt
            elif not isinstance(nxt, dict):
                logger.warning("Skipping state_delta path %r: %r is not a dict", dotted, key)
                ok = False
                break
            node = nxt
        if not ok:
            continue

        leaf = parts[-1]
        if isinstance(value, dict) and "__append__" in value:
            target = node.get(leaf)
            if target is None:
                target = []
                node[leaf] = target
            if not isinstance(target, list):
                logger.warning("Skipping __append__ at %r: target is not a list", dotted)
                continue
            target.append(value["__append__"])
        else:
            node[leaf] = value


def _parse_sim_response(content: str) -> dict | None:
    """Leniently extract the tool-sim's ``{"response", "state_delta"}`` JSON.

    Mirrors judge.py's ``_parse_judge_response`` recovery ladder: direct parse,
    then a ```` ```json ```` / bare ```` ``` ```` code block, then the first
    ``{ ... }`` brace boundary. Returns ``None`` when nothing parses (the caller
    then treats the raw text as the tool response with no state mutation).
    """
    try:
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, TypeError):
        pass

    code_block = re.search(r"```(?:json)?\s*\n(.*?)\n\s*```", content, re.DOTALL)
    if code_block:
        try:
            parsed = json.loads(code_block.group(1).strip())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(content[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    return None


def tool_to_json_schema(tool: dict) -> dict:
    """Convert a repo tool definition into an OpenAI-style function schema.

    Repo tools use a parameters LIST of {name, type, description, required, enum};
    LangChain ``bind_tools`` expects an OpenAI function with a JSON Schema
    ``parameters`` object (properties dict + required array).

    Args:
        tool: A scenario tool definition.

    Returns:
        An OpenAI-style ``{"type": "function", "function": {...}}`` dict.
    """
    properties: dict[str, dict] = {}
    required: list[str] = []

    for param in tool.get("parameters", []):
        name = param.get("name")
        if not name:
            continue
        raw_type = str(param.get("type", "string")).lower()
        json_type = raw_type if raw_type in _JSON_SCHEMA_TYPES else "string"

        prop: dict = {"type": json_type}
        if param.get("description"):
            prop["description"] = param["description"]
        if param.get("enum"):
            prop["enum"] = param["enum"]
        properties[name] = prop

        if param.get("required"):
            required.append(name)

    parameters_schema: dict = {
        "type": "object",
        "properties": properties,
    }
    if required:
        parameters_schema["required"] = required

    return {
        "type": "function",
        "function": {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "parameters": parameters_schema,
        },
    }


class SimulationRunner:
    """Runs multi-turn agent simulations."""

    def __init__(self, config: SimulationConfig | None = None):
        self.config = config or DEFAULT_SIMULATION

        # Validate the behavioral profile up front (issue #59): an unknown name
        # raises here, before any simulator client is built or paid call made,
        # rather than silently running cooperative while stamping a profile that
        # never applied.
        profile_instructions(self.config.user_sim_profile)

        # User simulator — generates realistic user turns. Provider comes from the
        # config (default "openai") so an override (issue #50) can route the sim to
        # a different family (e.g. Claude) through the existing registry.
        self._user_sim = create_model(
            ModelSpec(
                name="user_simulator",
                model_id=self.config.user_simulator_model,
                provider=self.config.user_simulator_provider,
                temperature=self.config.user_simulator_temperature,
            )
        )

        # Tool simulator — generates realistic tool responses.
        self._tool_sim = create_model(
            ModelSpec(
                name="tool_simulator",
                model_id=self.config.tool_simulator_model,
                provider=self.config.tool_simulator_provider,
                temperature=self.config.tool_simulator_temperature,
            )
        )

    @staticmethod
    def _usage_tokens(response) -> tuple[int, int]:
        """Extract (input_tokens, output_tokens) from a LangChain response.

        Returns ``(0, 0)`` when the provider returned no ``usage_metadata`` so a
        missing count never crashes a run — it just under-counts that call's
        contribution to the cost guard, which is the safe direction (the preflight
        estimate already over-states; a missing-usage provider only makes the
        actual look cheaper, never falsely tripping the cap).
        """
        usage = getattr(response, "usage_metadata", None)
        if not usage:
            return 0, 0
        return usage.get("input_tokens", 0) or 0, usage.get("output_tokens", 0) or 0

    def _accumulate_sim_tokens(self, response) -> None:
        """Add one simulator call's token usage to the per-run sim totals.

        Only ever called from within :meth:`run`, which initializes the counters
        before any simulator call. Uses ``getattr`` defaults so a test that drives
        a simulator helper directly (without going through ``run``) does not crash.
        """
        in_t, out_t = self._usage_tokens(response)
        self._sim_input_tokens = getattr(self, "_sim_input_tokens", 0) + in_t
        self._sim_output_tokens = getattr(self, "_sim_output_tokens", 0) + out_t

    @staticmethod
    def _state_progress(scenario: Scenario, world: dict | None) -> float | None:
        """Deterministic state-check pass fraction for ``world`` right now.

        Runs the same grader the post-run scorer uses (``score_state_changes``)
        against the run's *current* mutated world, so we can tell whether the
        scenario's goals are verifiably met at any point — independent of whether
        the user simulator thinks it is done. Returns the pass fraction in
        ``[0, 1]``, or ``None`` when the scenario has no ``ground_truth`` (state
        grading is inapplicable). Never raises: a grader hiccup must not sink a
        run, so any exception degrades to ``None``.
        """
        if scenario.ground_truth is None or world is None:
            return None
        try:
            result = score_state_changes(
                scenario.ground_truth, world, scenario.expected_state_changes
            )
        except Exception:
            logger.exception("State-progress grading failed for %s; recording None", scenario.id)
            return None
        return None if result is None else result["score"]

    @staticmethod
    def _coordination_verdict(
        scenario: Scenario,
        world: dict | None,
        agent_mutated_keys: set,
        user_mutated_keys: set,
    ) -> bool | None:
        """Deterministic dual-control coordination verdict (issue #58).

        Returns:
            - ``None`` when the scenario is not dual-control, has no ground_truth,
              or no user action fired (no coordination ever occurred to grade —
              the callers in :meth:`run` only call this when at least one fired).
            - ``True`` iff the agent COORDINATED correctly: the scenario's normal
              ``expected_state_changes`` ALL pass (the correct end state was
              reached given the user's concurrent actions) AND the attribution
              contract holds — the agent did NOT itself mutate any user-owned
              top-level key (no double-apply of what the user already did).

        Attribution is the subtle part of issue #58. The shared world is ONE
        world, but every mutation is tagged at application time: agent-side
        deltas (via the tool sim) record into ``agent_mutated_keys``, user-side
        deltas (the scripted user actions) into ``user_mutated_keys``. A key the
        user owned (it falls in a declared user_tool scope) that the AGENT also
        wrote is a coordination failure — the canonical "the agent re-applied the
        approval/update the customer already made" double-apply. Reusing the
        existing state grader for the end-state half means NO new scoring
        machinery for that part; the attribution half is a pure set check.

        Never raises: a grader hiccup degrades to None so a dual-control scenario
        can't sink a run.
        """
        dc = scenario.dual_control
        if dc is None or scenario.ground_truth is None or world is None:
            return None
        try:
            base = score_state_changes(
                scenario.ground_truth, world, scenario.expected_state_changes
            )
            base_ok = base is None or base["score"] >= 1.0
            # User-owned keys: anything in ANY declared user_tool's scope. The
            # agent must not have mutated one — that key is the user's to write.
            # We check against the declared SCOPE (not only what the user actually
            # touched) so the contract is "the agent stayed out of user territory"
            # regardless of which user action fired this run.
            user_owned: set = set()
            for tool in dc.user_tools.values():
                user_owned.update(tool.scope)
            # Defensive: also treat any key a user action actually wrote as owned.
            user_owned.update(user_mutated_keys)
            trespass = agent_mutated_keys & user_owned
            attribution_ok = not trespass
            return bool(base_ok and attribution_ok)
        except Exception:
            logger.exception("Coordination grading failed for %s; recording None", scenario.id)
            return None

    @staticmethod
    def _recovery_verdict(scenario: Scenario, world: dict | None) -> bool | None:
        """Deterministic recovery verdict for a probe-carrying scenario (#57).

        Returns:
            - ``None`` when the scenario has no recovery probe, or no
              ground_truth to grade against (recovery is unverifiable without a
              world).
            - ``True`` iff the agent recovered: the scenario's normal
              ``expected_state_changes`` ALL pass (the task got done despite the
              injected fault) AND the probe's ``recovery_assertions`` ALL pass
              (the agent did not act on the bad entity / contradiction the probe
              introduced). Both halves use the existing state grader — no new
              machinery.

        The two halves are graded separately and AND-ed so a partial — task done
        but acted on the wrong entity, or wrong entity avoided but task abandoned
        — counts as a NON-recovery. Never raises: a grader hiccup degrades to
        None so a probe can't sink a run.

        This helper only grades the final world — it cannot know whether the
        injection was ever delivered. Callers in :meth:`run` gate on the
        ``probe_fired`` flag and stamp ``recovered=None`` for a declared probe
        that never fired, so a conversation that ended before ``probe.turn``
        is never scored as if the fault had been injected.
        """
        probe = scenario.recovery_probe
        if probe is None or scenario.ground_truth is None or world is None:
            return None
        try:
            base = score_state_changes(
                scenario.ground_truth, world, scenario.expected_state_changes
            )
            base_ok = base is None or base["score"] >= 1.0
            if probe.recovery_assertions:
                extra = score_state_changes(scenario.ground_truth, world, probe.recovery_assertions)
                extra_ok = extra is not None and extra["score"] >= 1.0
            else:
                extra_ok = True
            return bool(base_ok and extra_ok)
        except Exception:
            logger.exception("Recovery grading failed for %s; recording None", scenario.id)
            return None

    def run(self, scenario: Scenario, agent_spec: ModelSpec) -> SimulationResult:
        """Run a complete multi-turn simulation for one scenario.

        Args:
            scenario: The test scenario with persona, goals, and tools.
            agent_spec: The model under test.

        Returns:
            SimulationResult with full conversation history and metrics.
        """
        agent = create_model(agent_spec)
        domain_config = DOMAIN_CONFIGS[scenario.domain]

        # Bind tools natively so the agent uses real function-calling, not a
        # bespoke JSON-in-text convention. Models without tool support fall back
        # to the unbound model (and the content regex below).
        tool_schemas = [tool_to_json_schema(t) for t in scenario.tools]
        try:
            agent_with_tools = agent.bind_tools(tool_schemas) if tool_schemas else agent
        except (NotImplementedError, AttributeError):
            logger.warning(
                "Model %s does not support bind_tools — falling back to "
                "content-embedded tool-call parsing",
                agent_spec.name,
            )
            agent_with_tools = agent

        turns: list[ConversationTurn] = []
        total_input_tokens = 0
        total_output_tokens = 0
        # Simulator-side tokens (user sim + tool sim), accumulated per-run for the
        # cost guard (issue #47). Reset every run() call so a runner reused across
        # scenarios does not leak one scenario's sim tokens into the next.
        self._sim_input_tokens = 0
        self._sim_output_tokens = 0
        total_latency_ms = 0.0
        completed = False
        resolved_model: str | None = None
        # End-condition bookkeeping (#32). Default is max-turns exhaustion; the
        # completion branch below overrides ended_by to "user_sim" if the sim
        # stops the conversation. state_progress_at_end is filled at the end.
        ended_by = "max_turns"
        state_progress_at_end: float | None = None
        premature_end = False

        # Stateful world for this run. Deep-copy so each reliability run starts
        # from a pristine ground_truth and mutations never leak across runs.
        # None for legacy scenarios -> stateless tool simulation (unchanged).
        world = copy.deepcopy(scenario.ground_truth) if scenario.ground_truth is not None else None

        # Dual control (issue #58). When the scenario declares a dual_control
        # block, the SIMULATED USER also acts on the shared world: each declared
        # user_action fires deterministically when its trigger is met, and its
        # state delta is applied through the SAME apply_state_delta the agent's
        # tools use (one world, one delta mechanism). Attribution stays sharp by
        # tagging mutations at application time: agent tool-sim deltas accumulate
        # into ``self._agent_mutated_keys`` (reset per run), user-action deltas
        # into ``user_mutated_keys`` below. ``pending`` is the list of not-yet-
        # fired actions (one-shot each); ``agent_tool_calls_seen`` feeds the
        # agent_called trigger. All inert for the single-control majority.
        dc = scenario.dual_control
        self._agent_mutated_keys: set = set()
        # Tool-sim parse-failure counter for THIS run (S3). Reset per run, exactly
        # like _agent_mutated_keys above; incremented in _simulate_tool_stateful
        # each time a stateful tool-sim response can't be parsed (so no state_delta
        # is applied), and read at SimulationResult construction below.
        self._tool_sim_parse_failures = 0
        # Coded-vs-LLM tool authority split for THIS run (#87 phase 1b). Reset per
        # run alongside the other per-run sim counters; incremented in
        # _simulate_tool_stateful (coded path vs LLM fallback) and read at
        # SimulationResult construction below.
        self._coded_transition_calls = 0
        self._llm_tool_sim_calls = 0
        self._llm_tool_sim_mutations = 0
        user_mutated_keys: set = set()
        user_actions_fired = 0
        user_actions_suppressed = 0
        pending_actions = dc.pending_actions() if dc is not None else []
        agent_tool_calls_seen: set = set()

        # Recovery probe (issue #57). ``probe_fired`` tracks whether the probe's
        # injected message was actually DELIVERED as a user turn — a probe is
        # only staged at the end of iteration probe.turn - 1, so a conversation
        # that ends earlier (early CONVERSATION_COMPLETE, max_turns < probe.turn,
        # an agent error) never fires it. Recovery is only graded when the probe
        # fired; otherwise ``recovered`` stays None (see SimulationResult).
        probe = scenario.recovery_probe
        probe_fired = False

        # System prompt no longer teaches a JSON tool-call convention — the model
        # uses native tool calling via the bound schemas.
        agent_system = (
            f"{domain_config['system_prompt']}\n\n"
            "Use the tools available to you whenever you need to look up or act on "
            "information. After receiving tool results, respond to the user. You may "
            "call tools multiple times in sequence before giving your final answer."
        )

        # Conversation history for the agent (LangChain message objects)
        agent_messages = [SystemMessage(content=agent_system)]
        current_user_message = scenario.initial_message

        # Counter so synthetic tool_call_ids are unique within a run when a
        # provider/fallback does not supply one.
        synthetic_id = 0

        for turn_num in range(self.config.max_turns):
            # Probe firing point: the probe's message was staged as
            # current_user_message at the END of iteration probe.turn - 1, so
            # reaching THIS iteration is the moment the agent actually sees the
            # fault. The flag is set at delivery, not at staging, so a probe
            # staged on the loop's final iteration (probe.turn == max_turns —
            # message staged but the loop ends before it is ever appended) does
            # not count as fired either.
            if probe is not None and probe.turn == turn_num:
                probe_fired = True
            # Record + append the user turn.
            turns.append(
                ConversationTurn(
                    turn_number=turn_num,
                    role="user",
                    content=current_user_message,
                )
            )
            agent_messages.append(HumanMessage(content=current_user_message))

            last_agent_content = ""

            # Inner agent<->tool loop for this user turn.
            for _ in range(MAX_TOOL_ROUNDS_PER_TURN + 1):
                start = time.perf_counter()
                try:
                    agent_response = agent_with_tools.invoke(agent_messages)
                except Exception as e:
                    logger.error("Agent error on turn %d: %s", turn_num, e)
                    return SimulationResult(
                        scenario_id=scenario.id,
                        domain=scenario.domain.value,
                        model_name=agent_spec.name,
                        turns=turns,
                        total_turns=len([t for t in turns if t.role == "agent"]),
                        total_latency_ms=total_latency_ms,
                        total_input_tokens=total_input_tokens,
                        total_output_tokens=total_output_tokens,
                        completed=False,
                        error=str(e),
                        final_world=world,
                        resolved_model=resolved_model,
                        ended_by="error",
                        state_progress_at_end=self._state_progress(scenario, world),
                        premature_end=False,
                        sim_input_tokens=self._sim_input_tokens,
                        sim_output_tokens=self._sim_output_tokens,
                        user_sim_model=self.config.user_simulator_model,
                        tool_sim_model=self.config.tool_simulator_model,
                        sim_profile=self.config.user_sim_profile,
                        # Dual control (issue #58): stamp whether this was a
                        # dual-control scenario, how many user actions fired
                        # before the error, and the coordination verdict ONLY if
                        # at least one fired (else None — no coordination to
                        # grade). False/0/None for the single-control majority.
                        dual_control=dc is not None,
                        user_actions_fired=user_actions_fired,
                        user_actions_suppressed=user_actions_suppressed,
                        coordination_ok=(
                            self._coordination_verdict(
                                scenario, world, self._agent_mutated_keys, user_mutated_keys
                            )
                            if user_actions_fired > 0
                            else None
                        ),
                        recovery_probe_kind=(
                            scenario.recovery_probe.kind if scenario.recovery_probe else None
                        ),
                        # Grade recovery only if the probe actually fired before
                        # the agent errored; a fault never injected is None.
                        recovered=(
                            self._recovery_verdict(scenario, world) if probe_fired else None
                        ),
                        probe_fired=probe_fired,
                        tool_sim_parse_failures=self._tool_sim_parse_failures,
                        coded_transition_calls=self._coded_transition_calls,
                        llm_tool_sim_calls=self._llm_tool_sim_calls,
                        llm_tool_sim_mutations=self._llm_tool_sim_mutations,
                    )
                agent_latency = (time.perf_counter() - start) * 1000
                total_latency_ms += agent_latency

                # Record the provider-reported model on first sight (LangChain
                # puts it in response_metadata as "model_name" or "model").
                if resolved_model is None:
                    meta = getattr(agent_response, "response_metadata", None) or {}
                    resolved_model = meta.get("model_name") or meta.get("model") or None

                agent_content = (
                    agent_response.content
                    if isinstance(agent_response.content, str)
                    else str(agent_response.content)
                )
                last_agent_content = agent_content

                # Token accounting for THIS invocation (inner rounds included).
                usage = getattr(agent_response, "usage_metadata", None)
                if usage:
                    turn_input = usage.get("input_tokens", 0)
                    turn_output = usage.get("output_tokens", 0)
                else:
                    turn_input = 0
                    turn_output = 0
                    if turn_num == 0:
                        logger.warning(
                            "No usage_metadata from %s — token counts and "
                            "cost estimates will be inaccurate",
                            agent_spec.name,
                        )
                total_input_tokens += turn_input
                total_output_tokens += turn_output

                # Extract native tool calls; fall back to content regex only if
                # the provider returned none.
                tool_calls_this_turn, used_fallback = self._extract_tool_calls(
                    agent_response, agent_content, turn_num
                )
                if used_fallback:
                    logger.warning(
                        "Native tool_calls empty for %s; recovered %d call(s) from "
                        "content via fallback regex (measure how often this fires)",
                        agent_spec.name,
                        len(tool_calls_this_turn),
                    )

                # Append the agent message to history. With native tool calls we
                # must preserve the AIMessage's tool_calls so ToolMessage replies
                # bind to the right call_id.
                if tool_calls_this_turn and not used_fallback:
                    agent_messages.append(agent_response)
                else:
                    agent_messages.append(AIMessage(content=agent_content))

                # Assign synthetic ids where the provider/fallback gave none.
                for tc in tool_calls_this_turn:
                    if not tc.tool_call_id:
                        synthetic_id += 1
                        tc.tool_call_id = f"call_{turn_num}_{synthetic_id}"

                # Transcript: AGENT turn FIRST (it issued the calls), then tools.
                turns.append(
                    ConversationTurn(
                        turn_number=turn_num,
                        role="agent",
                        content=agent_content,
                        tool_calls=tool_calls_this_turn,
                        latency_ms=agent_latency,
                        token_count=turn_input + turn_output,
                    )
                )

                if not tool_calls_this_turn:
                    break  # Agent produced a user-facing message; hand back to user.

                # Run each tool, append result to transcript AND agent history.
                for tc in tool_calls_this_turn:
                    # Dual control (issue #58): record agent tool-call names so an
                    # ``agent_called`` user-action trigger can react to them (e.g.
                    # the user approves the request right after the agent sends
                    # it). Inert for single-control runs.
                    agent_tool_calls_seen.add(tc.tool_name)
                    tool_result = self._simulate_tool(
                        tc, scenario.tools, world, scenario.domain.value
                    )
                    tc.result = tool_result
                    turns.append(
                        ConversationTurn(
                            turn_number=turn_num,
                            role="tool",
                            content=tool_result,
                            tool_call_id=tc.tool_call_id,
                        )
                    )
                    if used_fallback:
                        # No native tool_call_id on the AIMessage to bind to; feed
                        # the result back as a human-readable message instead.
                        agent_messages.append(
                            HumanMessage(content=f"Tool result for {tc.tool_name}:\n{tool_result}")
                        )
                    else:
                        agent_messages.append(
                            ToolMessage(content=tool_result, tool_call_id=tc.tool_call_id)
                        )
                # Loop again: agent now sees tool results and continues.
            else:
                # Inner loop exhausted without a tool-free agent message.
                logger.warning(
                    "Tool-round cap (%d) hit on turn %d for %s; handing back to user",
                    MAX_TOOL_ROUNDS_PER_TURN,
                    turn_num,
                    agent_spec.name,
                )

            # Dual control (issue #58): fire any pending user action whose trigger
            # is now met, BEFORE generating the next user turn. The next user turn
            # is turn_num + 1. Each action fires at most once (popped from
            # ``pending``); its state delta is applied through the SAME
            # apply_state_delta the agent's tools use — one shared world — and
            # every top-level key it writes is recorded USER-attributed. If the
            # action carries a ``user_message``, that scripted text becomes the
            # next user turn (the user voices the coordination signal, e.g. "I
            # just approved that"), replacing the user-sim's generated turn,
            # exactly like a recovery-probe injection. A silent action mutates the
            # world only; the user sim still speaks that turn.
            #
            # Delivery gate (the #74 fired-but-not-delivered lesson): an action
            # may only fire when at least one delivery turn remains (next_turn <
            # max_turns). On the loop's FINAL iteration the staged user turn would
            # never be delivered — the agent would never see the coordination
            # signal and would get no turn to act on it — so firing there would
            # apply the user's delta and grade coordination_ok for a harness-
            # timing reason, not an agent failure (e.g. an ``agent_called`` action
            # whose watched tool is first called on the last outer turn). Such an
            # action is treated as NOT fired: no delta is applied, it is not
            # counted in ``user_actions_fired`` (so coordination_ok stays None
            # when nothing else fired and the coordination-rate denominator stays
            # honest), and it is counted in ``user_actions_suppressed`` so the
            # trigger-met-but-undeliverable case is auditable on the row/artifact,
            # mirroring the recovery-probe ``probe_fired`` gate. ``after_turn``
            # triggers cannot normally reach this gate (USER_ACTION_TURN_MAX = 9
            # < the default max_turns = 10); ``agent_called`` can.
            if pending_actions and world is not None and turn_num + 1 >= self.config.max_turns:
                # This branch can only run on the loop's FINAL iteration: the loop
                # is ``range(self.config.max_turns)`` so ``turn_num <= max_turns-1``,
                # and the guard requires ``turn_num+1 >= max_turns`` i.e.
                # ``turn_num == max_turns-1``. The loop exits right after, so the
                # suppressed actions are never re-evaluated and we deliberately do
                # NOT rebuild ``pending_actions`` here (unlike the fire branch's
                # ``still_pending``). The assertion pins that invariant so a future
                # change to the loop bound can't silently make this branch leave a
                # suppressed-but-still-pending action to be re-fired next iteration.
                assert turn_num + 1 == self.config.max_turns, (
                    "dual-control suppression branch must be terminal "
                    f"(turn_num={turn_num}, max_turns={self.config.max_turns})"
                )
                for action in pending_actions:
                    if action_fires(
                        action,
                        next_user_turn=turn_num + 1,
                        agent_tool_calls_so_far=agent_tool_calls_seen,
                    ):
                        user_actions_suppressed += 1
                        logger.info(
                            "Dual-control user action SUPPRESSED (tool=%s, trigger=%s): "
                            "trigger met at user turn %d but no delivery turn remains "
                            "(max_turns=%d) for %s — treated as not fired",
                            action.tool,
                            action.trigger,
                            turn_num + 1,
                            self.config.max_turns,
                            scenario.id,
                        )
            elif pending_actions and world is not None:
                next_turn = turn_num + 1
                fired_message: str | None = None
                still_pending = []
                for action in pending_actions:
                    if fired_message is None and action_fires(
                        action,
                        next_user_turn=next_turn,
                        agent_tool_calls_so_far=agent_tool_calls_seen,
                    ):
                        user_actions_fired += 1
                        if action.state_delta:
                            apply_state_delta(world, action.state_delta)
                            for key in action.delta_paths():
                                user_mutated_keys.add(key)
                        logger.info(
                            "Dual-control user action fired (tool=%s, trigger=%s) "
                            "at user turn %d for %s",
                            action.tool,
                            action.trigger,
                            next_turn,
                            scenario.id,
                        )
                        if action.user_message:
                            fired_message = action.user_message
                    else:
                        still_pending.append(action)
                pending_actions = still_pending
                if fired_message is not None:
                    current_user_message = fired_message
                    continue

            # Recovery probe (issue #57): if this scenario carries a probe whose
            # turn is the NEXT user turn, inject the scripted perturbation verbatim
            # instead of asking the user simulator to generate it. The probe text
            # is identical for every model on every run — that determinism is the
            # point of a controlled fault. The user sim still drives every other
            # turn; only the probe turn is overridden. (turn_num is the index of
            # the turn just completed; the next user turn is turn_num + 1.)
            # This only STAGES the probe text; it counts as fired when the next
            # iteration actually delivers it (see the top of the loop).
            if probe is not None and probe.turn == turn_num + 1:
                logger.info(
                    "Injecting recovery probe (kind=%s) as user turn %d for %s",
                    probe.kind,
                    probe.turn,
                    scenario.id,
                )
                current_user_message = probe.injected_message()
                continue

            # Generate next user turn (or detect completion).
            user_response = self._simulate_user_turn(scenario, turns, last_agent_content)
            if CONVERSATION_COMPLETE in user_response:
                # The user sim has signaled it is DONE TALKING. It does NOT get
                # to declare the goals MET — that is the deterministic state
                # check's job (#32). Record that the sim ended the conversation
                # and the state-check progress at this exact moment; if progress
                # is below 1.0, this is a premature ending (the sim quit before
                # the goals were verifiably done) and must be visible in the
                # artifact rather than silently counted as a success.
                completed = True
                ended_by = "user_sim"
                state_progress_at_end = self._state_progress(scenario, world)
                premature_end = state_progress_at_end is not None and state_progress_at_end < 1.0
                if premature_end:
                    logger.warning(
                        "Premature ending: user sim ended %s with state progress "
                        "%.2f < 1.0 (sim satisfied, goals not verifiably met)",
                        scenario.id,
                        state_progress_at_end,
                    )
                break
            current_user_message = user_response

        # If the loop exhausted the turn budget without the sim signaling done,
        # ended_by stays "max_turns"; still record where the state check landed
        # so a run that ran out of turns is comparable to one the sim ended.
        if ended_by != "user_sim":
            state_progress_at_end = self._state_progress(scenario, world)

        return SimulationResult(
            scenario_id=scenario.id,
            domain=scenario.domain.value,
            model_name=agent_spec.name,
            turns=turns,
            total_turns=len([t for t in turns if t.role == "agent"]),
            total_latency_ms=total_latency_ms,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            completed=completed,
            final_world=world,
            resolved_model=resolved_model,
            ended_by=ended_by,
            state_progress_at_end=state_progress_at_end,
            premature_end=premature_end,
            sim_input_tokens=self._sim_input_tokens,
            sim_output_tokens=self._sim_output_tokens,
            user_sim_model=self.config.user_simulator_model,
            tool_sim_model=self.config.tool_simulator_model,
            sim_profile=self.config.user_sim_profile,
            # Dual control (issue #58): record whether this was a dual-control
            # scenario, how many user actions fired, and — ONLY when at least one
            # fired — the deterministic coordination verdict (correct end state
            # given the user's concurrent actions AND the agent never wrote a
            # user-owned path). A dual-control scenario where no action fired
            # (conversation ended before any trigger) keeps coordination_ok=None
            # so it cannot pollute the coordination rate with a verdict on a
            # coordination that never happened, and an action whose trigger was
            # met only on the final outer turn (no delivery turn remaining) is
            # counted in ``user_actions_suppressed`` instead of fired — see the
            # delivery gate in the loop. False/0/None for the single-control
            # majority.
            dual_control=dc is not None,
            user_actions_fired=user_actions_fired,
            user_actions_suppressed=user_actions_suppressed,
            coordination_ok=(
                self._coordination_verdict(
                    scenario, world, self._agent_mutated_keys, user_mutated_keys
                )
                if user_actions_fired > 0
                else None
            ),
            # Recovery probe (issue #57): record the declared kind, whether the
            # probe actually fired, and — ONLY for a fired probe — the
            # deterministic recovery verdict against the final world. A declared
            # probe that never fired (conversation ended before probe.turn)
            # keeps recovered=None so it cannot pollute recovery_rate with a
            # verdict on a fault that was never injected. kind/recovered are
            # None and probe_fired False for the (overwhelming) non-probe
            # majority.
            recovery_probe_kind=(scenario.recovery_probe.kind if scenario.recovery_probe else None),
            recovered=(self._recovery_verdict(scenario, world) if probe_fired else None),
            probe_fired=probe_fired,
            # Tool-sim parse-failure count (S3): how many stateful tool-sim
            # responses this run could not be parsed (so no state_delta applied).
            # >0 on a stateful scenario makes the state grade non-gradable.
            tool_sim_parse_failures=self._tool_sim_parse_failures,
            # Coded-vs-LLM tool authority split (#87 phase 1b).
            coded_transition_calls=self._coded_transition_calls,
            llm_tool_sim_calls=self._llm_tool_sim_calls,
            # Phase-3 spine-trust guard (#87): LLM-fallback mutations of the world.
            llm_tool_sim_mutations=self._llm_tool_sim_mutations,
        )

    def _extract_tool_calls(self, response, content: str, turn: int) -> tuple[list[ToolCall], bool]:
        """Extract tool calls from an agent response.

        Prefers LangChain native ``response.tool_calls`` (normalized across
        OpenAI/Anthropic/Google/OpenRouter). Falls back to parsing a
        ``{"tool_call": ...}`` object embedded in content ONLY when no native
        calls are present.

        Returns:
            (tool_calls, used_fallback) — ``used_fallback`` is True when the
            content regex produced the calls.
        """
        native = getattr(response, "tool_calls", None)
        if native:
            calls = [
                ToolCall(
                    turn=turn,
                    tool_name=tc.get("name", ""),
                    arguments=tc.get("args", {}) or {},
                    result="",
                    tool_call_id=tc.get("id") or "",
                )
                for tc in native
            ]
            return calls, False

        # Fallback: content-embedded JSON tool call (legacy protocol).
        fallback = self._extract_from_content(content, turn)
        return fallback, bool(fallback)

    def _extract_from_content(self, content: str, turn: int) -> list[ToolCall]:
        """Legacy fallback: parse {"tool_call": {...}} from response content."""
        calls: list[ToolCall] = []

        # Whole response is a single tool_call JSON object.
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict) and "tool_call" in parsed:
                tc = parsed["tool_call"]
                calls.append(
                    ToolCall(
                        turn=turn,
                        tool_name=tc.get("name", ""),
                        arguments=tc.get("arguments", {}),
                        result="",
                    )
                )
                return calls
        except (json.JSONDecodeError, TypeError):
            pass

        # Embedded tool_call objects.
        for match in _TOOL_CALL_RE.finditer(content):
            try:
                tc = json.loads(match.group(1))
                calls.append(
                    ToolCall(
                        turn=turn,
                        tool_name=tc.get("name", ""),
                        arguments=tc.get("arguments", {}),
                        result="",
                    )
                )
            except (json.JSONDecodeError, KeyError):
                continue

        return calls

    def _simulate_tool(
        self,
        tool_call: ToolCall,
        available_tools: list[dict],
        world: dict | None,
        domain: str | None = None,
    ) -> str:
        """Produce the tool result the agent sees, mutating the world if stateful.

        Three paths:

        - **Stateless (legacy)** — ``world`` is ``None`` (scenario has no
          ground_truth). The simulator invents a realistic response from the tool
          schema and args, exactly as before.
        - **Coded transition (v0.3, #87 phase 1b)** — ``world`` is set AND
          ``(domain, tool_name)`` has a registered deterministic transition. The
          coded function is the sole authority over the world mutation; the LLM is
          never called for this tool.
        - **Stateful LLM fallback (v0.2)** — ``world`` is set but the tool has no
          coded transition. The simulator is told to answer ONLY from the world
          and to return both the tool ``response`` and a ``state_delta``; only the
          ``response`` part is fed back to the agent.
        """
        tool_schema = next(
            (t for t in available_tools if t.get("name") == tool_call.tool_name),
            None,
        )

        if world is None:
            return self._simulate_tool_stateless(tool_call, tool_schema)
        return self._simulate_tool_stateful(tool_call, tool_schema, world, domain)

    def _simulate_tool_stateless(self, tool_call: ToolCall, tool_schema: dict | None) -> str:
        """Legacy stateless tool simulation (no ground-truth world)."""
        prompt = (
            "You are simulating a tool/API response. Generate a realistic, "
            "well-formed response for this tool call.\n\n"
            f"Tool: {tool_call.tool_name}\n"
            f"Arguments: {json.dumps(tool_call.arguments)}\n"
        )
        if tool_schema:
            prompt += f"Tool Schema: {json.dumps(tool_schema)}\n"
        prompt += (
            "\nRespond with ONLY the JSON response the tool would return. "
            "Make the data realistic but fictional."
        )

        response = self._tool_sim.invoke([HumanMessage(content=prompt)])
        self._accumulate_sim_tokens(response)
        return response.content if isinstance(response.content, str) else str(response.content)

    def _simulate_tool_stateful(
        self,
        tool_call: ToolCall,
        tool_schema: dict | None,
        world: dict,
        domain: str | None = None,
    ) -> str:
        """Stateful tool result: answer from + mutate the canonical world.

        Returns the serialized ``response`` part only (the agent never sees the
        ``state_delta`` or the world). Applies any ``state_delta`` to ``world`` in
        place so subsequent tool calls stay coherent.

        Authority over the world mutation is chosen per tool (#87 phase 1b):

        - If ``(domain, tool_name)`` has a registered **coded transition**, that
          deterministic pure function is the sole authority — it reads
          ``(args, world)`` and returns the same ``{response, state_delta}`` shape
          the LLM sim would, but as a byte-identical function of its inputs. The
          LLM is NOT called. This is what makes the graded world reproducible.
        - Otherwise the run falls back to the **LLM tool simulator** (the v0.2
          behavior): prompt the sim to answer from STATE and emit a state_delta.

        Both paths funnel through ``_commit_tool_result`` so the write-scope clamp,
        the delta application, and the dual-control attribution are identical
        regardless of who authored the delta.
        """
        # Coded-transition path (#87 phase 1b): deterministic, no LLM call.
        transition = get_transition(domain, tool_call.tool_name) if domain else None
        if transition is not None:
            self._coded_transition_calls = getattr(self, "_coded_transition_calls", 0) + 1
            parsed = transition(dict(tool_call.arguments or {}), world)
            # A coded transition's return is trusted COMPLETELY: its delta is never
            # counted as an LLM mutation, so the state grade stays gradable. That
            # makes a malformed return a SILENT mis-grade — a missing/typo'd
            # ``state_delta`` drops the mutation while the run is still graded, and a
            # missing ``response`` makes ``_commit_tool_result`` serialize the whole
            # ``{response, state_delta}`` dict back to the agent (leaking internal
            # state). Fail loud on a programmer error in a transition instead. The
            # corpus coverage + per-transition purity tests keep this inert, so it is
            # a defense-in-depth backstop, not a path real transitions hit. (Empty
            # ``state_delta`` {} is VALID — read-only / in-task-error shape.)
            if not (
                isinstance(parsed, dict)
                and isinstance(parsed.get("state_delta"), dict)
                and "response" in parsed
            ):
                raise ValueError(
                    f"Coded transition for {tool_call.tool_name!r} returned a malformed "
                    f"result; expected a dict with a dict 'state_delta' and a 'response', "
                    f"got: {parsed!r}"
                )
            return self._commit_tool_result(
                parsed, tool_schema, world, tool_call.tool_name, coded=True
            )

        # LLM fallback path (v0.2): the simulator authors the response + delta.
        self._llm_tool_sim_calls = getattr(self, "_llm_tool_sim_calls", 0) + 1
        prompt = (
            "You are simulating a tool/API backend that operates on a canonical "
            "STATE (the source of truth). Answer ONLY from STATE. Do not invent "
            "balances, IDs, or records not present in STATE. If the call should "
            "fail (unknown ID, insufficient funds, etc.), return a realistic error "
            "response and an empty state_delta.\n\n"
            f"Tool: {tool_call.tool_name}\n"
            f"Arguments: {json.dumps(tool_call.arguments)}\n"
        )
        if tool_schema:
            # Strip the `writes` authorization allow-list before showing the
            # schema to the sim: it only generates a state_delta (which is clamped
            # against `writes` regardless), so the write-authorization is not its
            # concern — mirrors UserTool.as_tool_schema dropping `scope`. Copy, so
            # the source dict the clamp reads downstream is unchanged.
            sim_schema = {k: v for k, v in tool_schema.items() if k != "writes"}
            prompt += f"Tool Schema: {json.dumps(sim_schema)}\n"
        prompt += (
            f"\nCURRENT STATE (JSON):\n{json.dumps(world)}\n\n"
            "Respond with ONLY a JSON object of the form:\n"
            '{"response": <what the tool returns to the caller>, '
            '"state_delta": <map of dotted-path -> new value, or {} if read-only>}\n'
            "Rules for state_delta:\n"
            '- Keys are dotted paths into STATE, e.g. "accounts.BUS-CHK-001.balance".\n'
            "- A plain value REPLACES the value at that path.\n"
            '- To append to a list, use {"__append__": <item>} as the value, e.g. '
            '"recurring_transfers": {"__append__": {"from": "BUS-CHK-001", ...}}.\n'
            "- Read-only calls (lookups) return an empty state_delta {}.\n"
            "- Apply each mutation exactly as the tool would (e.g. a transfer "
            "decrements the source and increments the destination)."
        )

        response = self._tool_sim.invoke([HumanMessage(content=prompt)])
        self._accumulate_sim_tokens(response)
        raw = response.content if isinstance(response.content, str) else str(response.content)

        parsed = _parse_sim_response(raw)
        if parsed is None:
            # No state_delta is applied, so the world is now missing whatever
            # mutation this call should have made. Count it (S3) so a stateful
            # scenario with a dropped mutation can be flagged non-gradable rather
            # than silently scored against an incomplete world. Guard getattr in
            # case a tool-sim is invoked outside run_scenario (e.g. a unit test).
            self._tool_sim_parse_failures = getattr(self, "_tool_sim_parse_failures", 0) + 1
            logger.warning(
                "Tool-sim output for %s unparseable; feeding raw text back, no state_delta",
                tool_call.tool_name,
            )
            return raw

        return self._commit_tool_result(
            parsed, tool_schema, world, tool_call.tool_name, coded=False
        )

    def _commit_tool_result(
        self,
        parsed: dict,
        tool_schema: dict | None,
        world: dict,
        tool_name: str,
        coded: bool,
    ) -> str:
        """Clamp, apply, attribute, and serialize a ``{response, state_delta}``.

        The single commit path shared by the coded-transition and LLM-fallback
        branches of ``_simulate_tool_stateful``. ``parsed`` is a dict carrying a
        ``state_delta`` (dotted-path -> new value, the format ``apply_state_delta``
        consumes) and a ``response`` (what the agent sees). Whoever authored the
        delta, it is funneled through the identical write-scope clamp, applier, and
        dual-control attribution, then the response is serialized and returned.

        ``coded`` says who authored ``parsed``: a deterministic coded transition
        (``True``) or the LLM tool-sim fallback (``False``). It governs the phase-3
        spine-trust guard only — when the LLM fallback applies a NON-EMPTY delta to
        the graded world (``coded=False`` and a post-clamp ``delta``), that mutation
        is AI-improvised and not reproducible, so it is counted into
        ``_llm_tool_sim_mutations``; ``is_state_gradable`` then nulls the state
        grade exactly as a parse failure does. A read-only LLM fallback (empty
        delta) does not taint the world and is not counted. Coded mutations are
        deterministic and never counted. Inert on the current corpus, where every
        tool is registered (the coverage test guarantees no LLM mutation), but a
        standing tripwire the moment a stateful scenario uses an unregistered tool.
        """
        delta = parsed.get("state_delta")
        if delta and isinstance(delta, dict):
            # Write-scope clamp (Option A, issue #58 determinism fix). When the
            # CALLED agent tool declares a ``writes`` allow-list (the top-level
            # state keys it is permitted to mutate), drop any delta path whose
            # top-level key is NOT in that list BEFORE applying it. For the LLM
            # path this catches a hallucinated stray write (e.g. simulating
            # ``open_support_ticket`` it invents a ``contact.email`` write); for the
            # coded path it is the documented backstop enforcing each transition's
            # declared scope. Without clamping a stray write lands in the world AND
            # in ``_agent_mutated_keys``, so the coordination verdict flags a FALSE
            # trespass (agent wrote a user-owned key). Clamping to the declared
            # scope makes both the world mutation and the attribution set a function
            # of the tool's contract. A tool that does NOT declare ``writes``
            # (writes is None) is NOT clamped — behavior is byte-identical to
            # before, which keeps the 92 single-control public scenarios (no tool
            # declares ``writes``) unchanged.
            writes = tool_schema.get("writes") if tool_schema else None
            if writes is not None:
                allowed = {str(k) for k in writes}
                in_scope: dict = {}
                for dotted, value in delta.items():
                    top = str(dotted).split(".", 1)[0]
                    if top in allowed:
                        in_scope[dotted] = value
                    else:
                        logger.warning(
                            "Tool %s produced out-of-scope write '%s' "
                            "(top-level key '%s' not in declared writes %s); dropping",
                            tool_name,
                            dotted,
                            top,
                            sorted(allowed),
                        )
                delta = in_scope

        if delta:
            if not coded:
                # Phase-3 spine-trust guard (#87): the LLM fallback just authored a
                # real mutation of the graded world. That mutation is unseeded and
                # not reproducible, so the state grade computed from this world can
                # no longer be trusted — count it so is_state_gradable nulls the
                # state grade (mirrors the parse-failure path). getattr default
                # keeps a direct unit-test call from crashing pre-run().
                self._llm_tool_sim_mutations = getattr(self, "_llm_tool_sim_mutations", 0) + 1
            apply_state_delta(world, delta)
            # Dual-control attribution (issue #58): record the top-level keys this
            # AGENT tool call mutated, so the coordination verdict can tell whether
            # the agent wrote into user-owned territory. After the write-scope
            # clamp above, every recorded key is one the called tool was permitted
            # to write, so the set is deterministic. Single-control runs never read
            # this set, so the bookkeeping is inert there. ``getattr`` default keeps
            # a direct call (in a unit test) from crashing when run() never
            # initialized the set.
            sink = getattr(self, "_agent_mutated_keys", None)
            if sink is not None and isinstance(delta, dict):
                for dotted in delta:
                    sink.add(str(dotted).split(".", 1)[0])

        tool_response = parsed.get("response", parsed)
        if isinstance(tool_response, str):
            return tool_response
        return json.dumps(tool_response)

    @staticmethod
    def _user_known_facts(scenario: Scenario) -> dict | None:
        """Customer-side slice of ground_truth the simulated user KNOWS.

        A real customer knows their own identity facts (customer id, SSN last4)
        and which accounts are theirs — without this, the user simulator invents
        values when the agent asks verification questions, the stateful tool sim
        rejects them against ground truth, and every scenario with an identity
        gate fails for harness reasons rather than agent reasons (found in the
        first live smoke run).

        Deliberately excluded: server-side state like ``verified`` flags,
        balances, and transaction details — the user should know who they are,
        not what the bank's systems currently say.
        """
        gt = scenario.ground_truth
        if not gt:
            return None
        known: dict = {}
        for key in ("customer", "contact", "user_profile"):
            block = gt.get(key)
            if isinstance(block, dict):
                known[key] = {k: v for k, v in block.items() if k != "verified"}
        accounts = gt.get("accounts")
        if isinstance(accounts, dict):
            known["your_accounts"] = {
                acct_id: (acct.get("type", "") if isinstance(acct, dict) else "")
                for acct_id, acct in accounts.items()
            }
        account = gt.get("account")
        if isinstance(account, dict):
            known["your_account"] = {
                k: account[k]
                for k in ("account_id", "company_name", "company", "name", "plan", "tier")
                if k in account
            }
        return known or None

    def _simulate_user_turn(
        self,
        scenario: Scenario,
        history: list[ConversationTurn],
        last_agent_response: str,
    ) -> str:
        """Generate the next user message based on persona and goals."""
        # Last 10 turns for context window efficiency
        history_text = "\n".join(f"[{t.role}] {t.content}" for t in history[-10:])

        prompt = (
            "You are simulating a user in a conversation with an AI agent.\n\n"
            f"Persona: {json.dumps(scenario.persona)}\n"
        )
        known = self._user_known_facts(scenario)
        if known:
            prompt += (
                "Facts you know about yourself and your accounts — answer from "
                "these EXACTLY when the agent asks (e.g. for identity "
                "verification). Never invent different values, and never "
                "volunteer them unprompted:\n"
                f"{json.dumps(known)}\n"
            )
        prompt += "Goals (pursue these naturally across the conversation):\n"
        for i, goal in enumerate(scenario.user_goals, 1):
            prompt += f"  {i}. {goal}\n"
        prompt += (
            f"\nConversation so far:\n{history_text}\n\n"
            f"Agent's last response:\n{last_agent_response}\n\n"
            "Instructions:\n"
            "- Respond naturally as this persona would\n"
            "- Pursue your remaining unmet goals\n"
            "- If ALL goals have been addressed satisfactorily, respond with "
            f'exactly "{CONVERSATION_COMPLETE}"\n'
            "- Do NOT be overly agreeable — push back if the agent's response is "
            "incomplete or unsatisfactory\n"
            "- Keep responses concise (1-3 sentences typically)\n"
        )

        # Behavioral profile block (issue #59 part 1). The cooperative default
        # maps to None and appends NOTHING — the prompt above stays byte-identical
        # to pre-profile behavior (pinned by a snapshot test). Non-cooperative
        # profiles layer a behavior block on top of the same persona/goals/facts;
        # {complete_token} lets a profile reference the end-of-conversation
        # signal without profiles.py importing this module.
        profile_block = profile_instructions(self.config.user_sim_profile)
        if profile_block is not None:
            prompt += "\n" + profile_block.format(complete_token=CONVERSATION_COMPLETE) + "\n"

        response = self._user_sim.invoke([HumanMessage(content=prompt)])
        self._accumulate_sim_tokens(response)
        return response.content if isinstance(response.content, str) else str(response.content)
