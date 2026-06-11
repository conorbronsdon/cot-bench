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
from eval.simulation.profiles import DEFAULT_SIM_PROFILE, profile_instructions

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
        user_mutated_keys: set = set()
        user_actions_fired = 0
        pending_actions = dc.pending_actions() if dc is not None else []
        agent_tool_calls_seen: set = set()

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
                        coordination_ok=(
                            self._coordination_verdict(
                                scenario, world, self._agent_mutated_keys, user_mutated_keys
                            )
                            if user_actions_fired > 0
                            else None
                        ),
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
                    tool_result = self._simulate_tool(tc, scenario.tools, world)
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
            if pending_actions and world is not None:
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
            # coordination that never happened. False/0/None for the
            # single-control majority.
            dual_control=dc is not None,
            user_actions_fired=user_actions_fired,
            coordination_ok=(
                self._coordination_verdict(
                    scenario, world, self._agent_mutated_keys, user_mutated_keys
                )
                if user_actions_fired > 0
                else None
            ),
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
        self, tool_call: ToolCall, available_tools: list[dict], world: dict | None
    ) -> str:
        """Use an LLM to generate a realistic tool response.

        Two paths:

        - **Stateless (legacy)** — ``world`` is ``None`` (scenario has no
          ground_truth). The simulator invents a realistic response from the tool
          schema and args, exactly as before.
        - **Stateful (v0.2)** — ``world`` is the run's mutable world dict. The
          simulator is told to answer ONLY from the world, and to return both the
          tool ``response`` and a ``state_delta`` describing any mutation. The
          delta is applied to ``world`` (so later calls see the change); only the
          ``response`` part is fed back to the agent — it never sees the delta or
          the world.
        """
        tool_schema = next(
            (t for t in available_tools if t.get("name") == tool_call.tool_name),
            None,
        )

        if world is None:
            return self._simulate_tool_stateless(tool_call, tool_schema)
        return self._simulate_tool_stateful(tool_call, tool_schema, world)

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
        self, tool_call: ToolCall, tool_schema: dict | None, world: dict
    ) -> str:
        """Stateful tool simulation: answer from + mutate the canonical world.

        Returns the serialized ``response`` part only (the agent never sees the
        ``state_delta`` or the world). Applies any parsed ``state_delta`` to
        ``world`` in place so subsequent tool calls stay coherent.
        """
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
            prompt += f"Tool Schema: {json.dumps(tool_schema)}\n"
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
            logger.warning(
                "Tool-sim output for %s unparseable; feeding raw text back, no state_delta",
                tool_call.tool_name,
            )
            return raw

        delta = parsed.get("state_delta")
        if delta:
            apply_state_delta(world, delta)
            # Dual-control attribution (issue #58): record the top-level keys this
            # AGENT tool call mutated, so the coordination verdict can tell whether
            # the agent wrote into user-owned territory. Single-control runs never
            # read this set, so the bookkeeping is inert there. ``getattr`` default
            # keeps a direct _simulate_tool_stateful call (in a unit test) from
            # crashing when run() never initialized the set.
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
