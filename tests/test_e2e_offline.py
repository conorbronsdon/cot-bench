"""Fully-offline end-to-end pipeline dry-run (issue #49).

This is the cheapest insurance against integration breakage in the paid rehearsal
run: it exercises the WHOLE pipeline in one pass with ZERO network calls and zero
API spend —

    run_eval.main()
        -> pre_registration.json  (written BEFORE any model call)
        -> SimulationRunner.run   (real agent<->tool<->user loop)
        -> multi-judge scoring     (real consensus / agreement / panel accounting)
        -> per-run artifacts        (real write_run_artifact)
        -> results parquet + run_manifest.json
    aggregate_results.compute_leaderboard / main()
        -> leaderboard.json        (alpha, pass^k, judge_deltas, length_bias,
                                     premature_end_rate, holdout_gap, rank bands)
    check_publish_ready.check_publish_ready()
        -> publish gate exit codes (complete vs models_failed)

Every other test in the suite pins one unit. This one pins that the units still
COMPOSE — that the row schema run_eval writes is the schema aggregate reads, that
the artifact carries every field calibration expects, and that the manifest the
publish gate reads is the manifest run_eval writes.

How "offline" is enforced
--------------------------
The only places the real pipeline reaches the network are:

  * ``eval.simulation.runner.create_model`` — builds the agent, user simulator,
    and tool simulator LangChain clients.
  * ``eval.scoring.judge._call_judge_api`` — the single judge HTTP call.

Both are monkeypatched to deterministic fakes. As a belt-and-suspenders guard we
also patch ``eval.providers.registry.create_model`` to raise (so any code path
that bypassed the runner seam would blow up loudly rather than silently dial out)
and assert ``_call_judge_api`` is never reached for a real client. No fake makes
an HTTP call; the fakes are plain in-memory objects.

The fakes
---------
* ``FakeAgentChatModel`` — a ``BaseChatModel`` (so ``bind_tools`` works) that, on
  its FIRST turn for a scenario, emits the tool calls scripted for that scenario
  (driven off a per-scenario plan keyed by the bound tool names), then on the
  next turn returns a tool-free user-facing message so the runner hands back to
  the user. It mutates nothing itself — the tool simulator applies the state
  deltas — exactly like a real agent.
* ``FakeToolSim`` — returns ``{"response", "state_delta"}`` JSON. The deltas are
  scripted so the deterministic state grader produces a NON-TRIVIAL partial score
  (some assertions pass, some are deliberately left unmet), which is what makes
  the state column meaningful rather than 0.0 or 1.0.
* ``FakeUserSim`` — ends conversations BOTH ways across scenarios: normally
  (emits ``[CONVERSATION_COMPLETE]`` once the agent has spoken) and prematurely
  (ends while the state check is still < 1.0, exercising premature_end), plus one
  scenario it lets run to the max-turns budget.
* ``FakeJudge`` — returns a parseable COMBINED-rubric JSON for two judges and an
  UNPARSEABLE body for the third on one scenario, so the panel parse-failure
  accounting (n_judges_valid, parse_failures, degraded) is exercised end to end.

All content authored here is SYNTHETIC. The two holdout scenarios are dummies in
the real v0.2 schema; no real holdout content appears in this repo.
"""

import json
import re
import zlib
from pathlib import Path

import pandas as pd
import pytest
from langchain_core.messages import AIMessage, HumanMessage

import scripts.aggregate_results as aggregate_results
import scripts.run_eval as run_eval
from eval.config import Domain
from eval.pre_registration import PRE_REGISTRATION_FILENAME
from eval.providers.null_agent import NULL_AGENT_NAME
from eval.simulation.runner import Scenario
from scripts.check_publish_ready import check_publish_ready

# A small, real, public subset (loaded from data/scenarios) — one per domain.
REAL_BANKING_ID = "banking_adaptive_tool_use_0001"
REAL_CS_ID = "cs_adaptive_tool_use_0001"

# ---------------------------------------------------------------------------
# Scenario plans: which tools the fake agent calls and which state deltas the
# fake tool-sim applies. Keyed by scenario id. Each plan is engineered so the
# deterministic state grader lands on a known PARTIAL score (not 0, not 1) for
# the real scenarios, so the state column is non-trivial in the parquet.
# ---------------------------------------------------------------------------

# Banking_0001 has 5 expected_state_changes:
#   1 accounts.BUS-CHK-001.balance increased_by 2500
#   2 accounts.BUS-SAV-002.balance decreased_by 2500
#   3 recurring_transfers contains {...weekly...}
#   4 fraud_cases contains {TXN-7781, DIGITAL_SVC_LLC}
#   5 customer.verified == true
# Plan: satisfy the transfer (1+2), verification (5), and fraud report (4) but
# deliberately SKIP the recurring transfer (3). -> 4/5 = 0.8 state score.
_BANKING_PLAN = {
    "tool_calls": [
        (
            "verify_customer_identity",
            {
                "customer_id": "CUST-88421",
                "verification_method": "ssn_last4",
                "verification_value": "4417",
            },
        ),
        (
            "initiate_transfer",
            {"from_account_id": "BUS-SAV-002", "to_account_id": "BUS-CHK-001", "amount": 2500.0},
        ),
        (
            "report_suspicious_transaction",
            {
                "account_id": "BUS-CHK-001",
                "transaction_id": "TXN-7781",
                "reason": "Unknown DIGITAL_SVC_LLC charge",
            },
        ),
    ],
    # tool name -> state_delta the fake tool-sim returns for it.
    "deltas": {
        "verify_customer_identity": {"customer.verified": True},
        "initiate_transfer": {
            "accounts.BUS-CHK-001.balance": 8420.55 + 2500.0,
            "accounts.BUS-SAV-002.balance": 15300.00 - 2500.0,
        },
        "report_suspicious_transaction": {
            "fraud_cases": {
                "__append__": {"transaction_id": "TXN-7781", "reason": "DIGITAL_SVC_LLC unknown"}
            }
        },
    },
}

# CS_0001 has 2 expected_state_changes (meetings contains, escalations contains).
# Plan: satisfy the meeting but SKIP the escalation. -> 1/2 = 0.5 state score.
_CS_PLAN = {
    "tool_calls": [
        ("schedule_meeting", {"customer_id": "CUST-CS-1", "topic": "QBR"}),
    ],
    "deltas": {
        # Match whatever the real cs_0001 "meetings contains" assertion looks for
        # loosely: append a rich dict so a partial-dict match is likely to hit.
        # If it does not match, state score is simply 0.0 for that scenario —
        # still a valid non-trivial e2e exercise; we assert state is present, not
        # a specific value, for CS.
        "schedule_meeting": {
            "meetings": {
                "__append__": {"customer_id": "CUST-CS-1", "type": "QBR", "scheduled": True}
            }
        },
    },
}

# Synthetic holdout scenarios: real v0.2 schema, empty expected_state_changes
# (the no-unauthorized-mutation contract). The fake agent makes NO tool calls for
# these, so the world is unchanged -> state score 1.0 (correctly does nothing).
_HOLDOUT_PLAN = {"tool_calls": [], "deltas": {}}

_PLANS = {
    REAL_BANKING_ID: _BANKING_PLAN,
    REAL_CS_ID: _CS_PLAN,
}


# Scenarios on which the user sim ends PREMATURELY (emits COMPLETE while state
# check is still < 1.0). Banking lands at 0.8 state -> premature. CS we let end
# normally. One holdout we let run to max_turns to exercise ended_by=="max_turns".
_PREMATURE_END_SCENARIOS = {REAL_BANKING_ID}
_RUN_TO_MAX_TURNS = set()  # filled per-test for the max-turns assertion


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _Recorder:
    """Shared call recorder so tests can order-assert pre-registration vs calls."""

    def __init__(self):
        self.model_calls = []  # spec names / roles, in creation order
        self.agent_invokes = 0
        self.tool_invokes = 0
        self.user_invokes = 0
        self.judge_calls = 0
        self.prereg_written_before_first_call = None


def _scenario_id_from_messages(messages) -> str | None:
    """Best-effort recover which scenario an agent invocation belongs to.

    The runner seeds the agent with a domain system prompt + the scenario's
    initial_message as the first HumanMessage. We match on the initial message
    text, which is unique per scenario in our subset.
    """
    for m in messages:
        if isinstance(m, HumanMessage) and isinstance(m.content, str):
            for sid, init in _INITIAL_MESSAGES.items():
                if init and init in m.content:
                    return sid
    return None


# Populated by the fixture once scenarios are loaded.
_INITIAL_MESSAGES: dict[str, str] = {}
# Per-scenario marker string GUARANTEED to appear in the user-sim prompt on every
# call (the user sim dumps persona + all goals every turn — unlike the rolling
# 10-turn transcript window, these never scroll out). Used to identify the
# scenario robustly across long conversations (e.g. the max-turns case).
_PROMPT_MARKERS: dict[str, str] = {}


def _scenario_from_prompt(prompt: str) -> str | None:
    """Identify the scenario from a user-sim prompt via its stable marker."""
    for sid, marker in _PROMPT_MARKERS.items():
        if marker and marker in prompt:
            return sid
    return None


class FakeAgentChatModel:
    """Deterministic agent that calls scripted tools then speaks (no network).

    Not a true BaseChatModel subclass — the runner only needs ``.invoke`` and
    ``.bind_tools`` and reads ``.content`` / ``.tool_calls`` / ``.usage_metadata``
    / ``.response_metadata`` off the response. Keeping it a plain object avoids
    pydantic field plumbing while matching the duck-typed interface exactly.
    """

    def __init__(self, recorder: _Recorder, model_name: str):
        self.recorder = recorder
        self.model_name = model_name
        self._bound_tool_names: set[str] = set()
        # Per-scenario: how many times we've been invoked (so turn 0 = tool calls,
        # later = a plain user-facing message to hand back to the user).
        self._invoke_counts: dict[str, int] = {}

    def bind_tools(self, tools, **kwargs):
        self._bound_tool_names = {
            t.get("function", {}).get("name") for t in tools if isinstance(t, dict)
        }
        return self

    def invoke(self, messages, **kwargs):
        self.recorder.agent_invokes += 1
        sid = _scenario_id_from_messages(messages)
        count = self._invoke_counts.get(sid, 0)
        self._invoke_counts[sid] = count + 1

        plan = _PLANS.get(sid, _HOLDOUT_PLAN)
        # First agent turn for this scenario: emit the scripted tool calls (only
        # those that are actually bound, so we never invent an unbound tool).
        if count == 0 and plan["tool_calls"]:
            tool_calls = []
            for i, (name, args) in enumerate(plan["tool_calls"]):
                if self._bound_tool_names and name not in self._bound_tool_names:
                    continue
                tool_calls.append({"name": name, "args": args, "id": f"call_{name}_{i}"})
            if tool_calls:
                return AIMessage(
                    content="",
                    tool_calls=tool_calls,
                    usage_metadata={"input_tokens": 120, "output_tokens": 40, "total_tokens": 160},
                    response_metadata={"model_name": self.model_name},
                )
        # Otherwise: a plain user-facing reply (no tool calls) -> runner hands
        # back to the user sim. Length scales a little so length-bias has variance.
        reply = "Here is what I did for you. " * (3 if sid == REAL_BANKING_ID else 1)
        return AIMessage(
            content=reply,
            usage_metadata={
                "input_tokens": 80,
                "output_tokens": 30 if sid == REAL_BANKING_ID else 10,
                "total_tokens": 110,
            },
            response_metadata={"model_name": self.model_name},
        )


class FakeToolSim:
    """Tool simulator: returns {"response", "state_delta"} scripted per tool."""

    def __init__(self, recorder: _Recorder):
        self.recorder = recorder

    def bind_tools(self, tools, **kwargs):  # never used, but keep the interface
        return self

    def invoke(self, messages, **kwargs):
        self.recorder.tool_invokes += 1
        prompt = messages[0].content if messages else ""
        # Recover the tool name from the prompt the runner builds ("Tool: <name>").
        tool_name = ""
        for line in prompt.splitlines():
            if line.startswith("Tool: "):
                tool_name = line[len("Tool: ") :].strip()
                break
        delta = {}
        for plan in (_BANKING_PLAN, _CS_PLAN):
            if tool_name in plan["deltas"]:
                delta = plan["deltas"][tool_name]
                break
        body = {"response": {"ok": True, "tool": tool_name}, "state_delta": delta}
        return AIMessage(
            content=json.dumps(body),
            usage_metadata={"input_tokens": 50, "output_tokens": 20, "total_tokens": 70},
        )


class FakeUserSim:
    """User simulator: ends normally or prematurely depending on the scenario.

    The runner calls this after each agent turn. We end the conversation (emit
    CONVERSATION_COMPLETE) as soon as the agent has produced a user-facing reply,
    EXCEPT for scenarios in ``_RUN_TO_MAX_TURNS`` which we keep talking on so the
    outer turn budget is exhausted (ended_by == "max_turns").
    """

    def __init__(self, recorder: _Recorder):
        self.recorder = recorder

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages, **kwargs):
        self.recorder.user_invokes += 1
        prompt = messages[0].content if messages else ""
        sid = _scenario_from_prompt(prompt)
        if sid in _RUN_TO_MAX_TURNS:
            return AIMessage(
                content="Please continue, I have more questions.",
                usage_metadata={"input_tokens": 30, "output_tokens": 10, "total_tokens": 40},
            )
        # Normal/premature end: the distinction (premature vs clean) is decided by
        # the deterministic state-check progress at end, NOT by the sim — exactly
        # the #32 decoupling. We just signal "done talking".
        from eval.simulation.runner import CONVERSATION_COMPLETE

        return AIMessage(
            content=f"Great, that's all. {CONVERSATION_COMPLETE}",
            usage_metadata={"input_tokens": 30, "output_tokens": 5, "total_tokens": 35},
        )


def _make_fake_create_model(recorder: _Recorder):
    """Return a create_model replacement routing by spec role.

    user_simulator / tool_simulator are recognized by their fixed ModelSpec
    names; everything else is a contestant agent (incl. the null agent, which we
    let through to the REAL null-agent factory so its do-nothing behavior — and
    near-zero scoring — is exercised by the same e2e path).
    """

    def fake_create_model(spec):
        if spec.name == "user_simulator":
            recorder.model_calls.append("user_simulator")
            return FakeUserSim(recorder)
        if spec.name == "tool_simulator":
            recorder.model_calls.append("tool_simulator")
            return FakeToolSim(recorder)
        if spec.provider == "null":
            # Use the REAL deterministic null agent (no network) so its scoring is
            # validated through the genuine code path.
            from eval.providers.null_agent import create_null_agent

            recorder.model_calls.append("null-agent")
            return create_null_agent(spec)
        recorder.model_calls.append(f"agent:{spec.name}")
        return FakeAgentChatModel(recorder, spec.model_id)

    return fake_create_model


def _make_fake_judge(recorder: _Recorder):
    """Return a _call_judge_api replacement returning combined-rubric JSON.

    Two judges return well-formed combined JSON; the third (opus) returns an
    UNPARSEABLE body on the banking scenario so the parse-failure accounting is
    exercised. Scores are derived deterministically from the transcript length so
    they vary across scenarios/models (gives alpha + length-bias something to
    chew on) but never call out to a model.
    """

    def fake_call_judge_api(judge, system_prompt, rubric_prompt):
        recorder.judge_calls += 1
        # Deterministic synthetic usage for the 3-tuple signature (#47); non-zero
        # so the cost accumulator path is exercised end to end.
        usage = (len(rubric_prompt) // 4, 120)
        # Banking transcripts are longer (the agent speaks 3x); use a crude proxy
        # for "this is the banking scenario" so the opus judge can parse-fail there.
        is_banking = REAL_BANKING_ID in rubric_prompt or "BUS-CHK-001" in rubric_prompt
        if judge.name == "Claude Opus 4.6" and is_banking:
            # Deliberately unparseable -> parse_failed for BOTH dimensions.
            return ("the verdict is: pretty good honestly, no json here", judge.model_id, usage)
        # Score varies by judge so inter-judge alpha is non-degenerate.
        base = {"Kimi K2.6": 0.7, "GLM-4.6": 0.6, "Claude Opus 4.6": 0.8}.get(judge.name, 0.65)
        body = {
            "task_completion": {
                "goal_scores": [],
                "overall_score": base,
                "overall_reasoning": "fake",
            },
            "tool_selection": {
                "tool_call_scores": [],
                "missed_tool_calls": [],
                "overall_score": round(base - 0.1, 4),
                "overall_reasoning": "fake",
            },
        }
        # The real scenarios carry rubric_criteria (#54), so the strict parser
        # requires a verdict block whose id set matches the prompt's criteria
        # EXACTLY — pull the ids from the appended criteria section. met varies
        # deterministically by judge+criterion (crc32, not salted hash()) so the
        # criterion-informed scores stay non-degenerate across the panel.
        _, _, criteria_section = rubric_prompt.partition("# Scenario-Specific Rubric Criteria")
        crit_ids = re.findall(r"^- \[([^\]]+)\]", criteria_section, flags=re.MULTILINE)
        if crit_ids:
            body["rubric_criteria"] = [
                {
                    "id": cid,
                    "met": zlib.crc32(f"{judge.name}:{cid}".encode()) % 10 < 7,
                    "evidence": "fake",
                }
                for cid in crit_ids
            ]
        return (json.dumps(body), judge.model_id, usage)

    return fake_call_judge_api


def _exploding_create_model(spec):
    raise AssertionError(
        f"registry.create_model was called for {spec!r} — the e2e test must route "
        "ALL model creation through the runner seam (offline guard tripped)."
    )


# ---------------------------------------------------------------------------
# Fixture: load the real subset + write a synthetic holdout, wire all fakes.
# ---------------------------------------------------------------------------


@pytest.fixture
def offline_pipeline(tmp_path, monkeypatch):
    """Wire run_eval + aggregate to run fully offline over a real+synthetic set.

    Yields a context object the tests drive (results_dir, recorder, a runner that
    invokes main() with given argv).
    """
    repo_root = Path(run_eval.__file__).resolve().parents[1]

    # Load the real public subset straight from data/scenarios (the genuine
    # loader + schema), then monkeypatch load_scenarios to return only our subset.
    banking = _load_real(repo_root, Domain.BANKING, REAL_BANKING_ID)
    cs = _load_real(repo_root, Domain.CUSTOMER_SUCCESS, REAL_CS_ID)
    subset = {Domain.BANKING: [banking], Domain.CUSTOMER_SUCCESS: [cs]}

    _INITIAL_MESSAGES.clear()
    _PROMPT_MARKERS.clear()
    for sc in (banking, cs):
        _INITIAL_MESSAGES[sc.id] = sc.initial_message
        # First user goal is unique per scenario and always rendered in full in
        # the user-sim prompt's Goals section (never truncated by the turn window).
        _PROMPT_MARKERS[sc.id] = sc.user_goals[0]

    def fake_load_scenarios(domain):
        return list(subset.get(domain, []))

    monkeypatch.setattr(run_eval, "load_scenarios", fake_load_scenarios)
    # Tracing off (no spans.jsonl side effects, keeps it fast). get_tracer returns
    # None, so the post-hoc trace emitters must be no-ops too (they would otherwise
    # dereference the None tracer).
    monkeypatch.setattr(run_eval, "init_tracing", lambda **kw: None)
    monkeypatch.setattr(run_eval, "get_tracer", lambda: None)
    monkeypatch.setattr(run_eval, "trace_agent_turn", lambda *a, **kw: None)
    monkeypatch.setattr(run_eval, "trace_judge_evaluation", lambda *a, **kw: None)
    monkeypatch.setattr(run_eval, "_trace_agent_turns", lambda *a, **kw: None)

    recorder = _Recorder()

    # Patch the runner seam (agent/user/tool) and the judge seam.
    monkeypatch.setattr("eval.simulation.runner.create_model", _make_fake_create_model(recorder))
    monkeypatch.setattr("eval.scoring.judge._call_judge_api", _make_fake_judge(recorder))
    # Belt-and-suspenders offline guard: any path that bypasses the runner seam
    # and hits the registry directly must explode, not dial out.
    monkeypatch.setattr("eval.providers.registry.create_model", _exploding_create_model)

    # Order guard: assert pre_registration is on disk before the first fake call.
    real_write = run_eval.write_pre_registration

    def watched_write(results_dir, pre_registration):
        path = real_write(results_dir, pre_registration)
        # No model/judge call may have happened yet.
        recorder.prereg_written_before_first_call = (
            recorder.agent_invokes == 0
            and recorder.tool_invokes == 0
            and recorder.user_invokes == 0
            and recorder.judge_calls == 0
            and Path(path).exists()
        )
        return path

    monkeypatch.setattr(run_eval, "write_pre_registration", watched_write)

    results_dir = tmp_path / "results"
    results_dir.mkdir()

    # Synthetic holdout dir (2 dummy scenarios, real schema, synthetic content).
    holdout_root = _write_synthetic_holdout(tmp_path)

    class Ctx:
        def __init__(self):
            self.results_dir = results_dir
            self.holdout_root = holdout_root
            self.recorder = recorder

        def run_main(self, extra_argv, stamp="20260610_120000"):
            output = results_dir / f"results_{stamp}.parquet"
            argv = [
                "run_eval",
                "--domains",
                "banking",
                "customer_success",
                "--reliability-runs",
                "2",
                "--parallel-models",
                "1",
                "--output",
                str(output),
                *extra_argv,
            ]
            monkeypatch.setattr("sys.argv", argv)
            run_eval.main()
            return output

        def aggregate(self):
            # Point aggregate at OUR results dir (it globs RESULTS_DIR for the
            # latest parquet), then run the real compute path.
            monkeypatch.setattr(aggregate_results, "RESULTS_DIR", results_dir)
            df = pd.read_parquet(sorted(results_dir.glob("results_*.parquet"))[-1])
            return aggregate_results.compute_leaderboard(df), df

    yield Ctx()


def _load_real(repo_root: Path, domain: Domain, scenario_id: str) -> Scenario:
    path = repo_root / "data" / "scenarios" / domain.value / f"{scenario_id}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return run_eval._scenario_from_dict(data, domain, holdout=False)


_DUMMY_HOLDOUT = {
    "id": "banking_adaptive_tool_use_9999_e2e000",
    "category": "adaptive_tool_use",
    "schema_version": "0.2",
    "authorship": {"author_model": "human-handwritten"},
    "persona": {
        "name": "Synthetic E2E",
        "age": 33,
        "occupation": "tester",
        "personality_traits": ["synthetic"],
        "tone": "neutral",
        "detail_level": "low",
        "background": "Fake persona for the e2e test.",
    },
    "user_goals": ["ask a harmless question", "say thanks"],
    "tools": [{"name": "noop", "description": "does nothing", "parameters": []}],
    "initial_message": "Hi, this is a synthetic E2E holdout opener number ",
    "difficulty": "low",
    # Empty assertions = no-unauthorized-mutation contract; the do-nothing agent
    # for holdout (no tool calls) leaves the world unchanged -> state 1.0.
    "ground_truth": {"accounts": {"X": {"balance": 1.0}}},
    "expected_state_changes": [],
}


def _write_synthetic_holdout(tmp_path) -> Path:
    root = tmp_path / "holdout"
    dom = root / "banking"
    dom.mkdir(parents=True)
    for i in range(2):
        data = dict(_DUMMY_HOLDOUT)
        data["id"] = f"banking_adaptive_tool_use_9999_e2e{i:03d}"
        data["initial_message"] = _DUMMY_HOLDOUT["initial_message"] + str(i)
        (dom / f"{data['id']}.json").write_text(json.dumps(data), encoding="utf-8")
        _INITIAL_MESSAGES[data["id"]] = data["initial_message"]
        # Holdout markers need to be UNIQUE per dummy so the two holdouts don't
        # collide; the per-dummy initial_message (with its index suffix) is unique.
        _PROMPT_MARKERS[data["id"]] = data["initial_message"]
    return root


# ===========================================================================
# Tests
# ===========================================================================


class TestEndToEndOffline:
    """One real run through main() + aggregate + publish gate, then assert the
    full surface produced. Most assertions share this single run via the class
    fixture to keep runtime well under the 60s budget."""

    @pytest.fixture(autouse=True)
    def _run_once(self, offline_pipeline):
        self.ctx = offline_pipeline
        # Real public subset (2 scenarios) + synthetic holdout (2) over 2 models,
        # 2 reliability runs each. Small enough to be fast, big enough that every
        # aggregation surface has >1 unit.
        self.output = self.ctx.run_main(
            [
                "--models",
                "GPT-5.5",
                "Claude Sonnet 4.6",
                "--holdout-dir",
                str(self.ctx.holdout_root),
            ]
        )
        self.leaderboard, self.df = self.ctx.aggregate()

    # --- pre-registration ----------------------------------------------------

    def test_pre_registration_written_before_any_model_call(self):
        assert self.ctx.recorder.prereg_written_before_first_call is True
        prereg_path = self.ctx.results_dir / PRE_REGISTRATION_FILENAME
        assert prereg_path.exists()

    def test_pre_registration_public_index_and_hash(self):
        reg = json.loads((self.ctx.results_dir / PRE_REGISTRATION_FILENAME).read_text("utf-8"))
        # Public scenario_set: the 2 real public scenarios, WITH index + per-scenario hash.
        assert reg["scenario_set"]["n_scenarios"] == 2
        assert len(reg["scenario_set"]["sha256"]) == 64
        ids = set(
            reg["scenario_set"]["scenario_ids_by_domain"]["banking"]
            + reg["scenario_set"]["scenario_ids_by_domain"]["customer_success"]
        )
        assert ids == {REAL_BANKING_ID, REAL_CS_ID}
        for entry in reg["scenario_set"]["scenario_index"]:
            assert len(entry["sha256"]) == 64
            assert {"domain", "scenario_id", "sha256"} <= set(entry)

    def test_pre_registration_holdout_block_hash_count_only(self):
        reg = json.loads((self.ctx.results_dir / PRE_REGISTRATION_FILENAME).read_text("utf-8"))
        block = reg["holdout_set"]
        assert block["n_scenarios"] == 2
        assert len(block["sha256"]) == 64
        # Privacy invariant: NO ids/index, and the dummy id never leaks anywhere.
        assert "scenario_index" not in block
        assert "scenario_ids_by_domain" not in block
        assert "9999" not in json.dumps(reg["holdout_set"])
        assert "e2e0" not in json.dumps(reg["holdout_set"])

    # --- artifacts -----------------------------------------------------------

    def test_artifacts_carry_full_field_surface(self):
        run_id = self.output.stem
        art_root = self.ctx.results_dir / "artifacts" / run_id
        files = list(art_root.rglob("*.json"))
        # 2 public + 2 holdout scenarios * 2 models * 2 runs = 16.
        assert len(files) == 16
        payload = json.loads(files[0].read_text("utf-8"))
        for key in (
            "scenario_id",
            "model",
            "run_index",
            "domain",
            "category",
            "holdout",
            "transcript",
            "judges",
            "sim_meta",
        ):
            assert key in payload, key
        sm = payload["sim_meta"]
        for key in (
            "completed",
            "ended_by",
            "state_progress_at_end",
            "premature_end",
            "resolved_model",
        ):
            assert key in sm, key
        # Both judge dimensions present, each a list of per-judge records.
        assert set(payload["judges"]) == {"task_completion", "tool_selection"}
        assert isinstance(payload["judges"]["task_completion"], list)

    def test_artifact_records_judge_parse_failure(self):
        # The opus judge parse-fails on the banking scenario; that must be visible
        # in the artifact's judge list (parse_failed=True kept for transparency).
        run_id = self.output.stem
        art_root = self.ctx.results_dir / "artifacts" / run_id
        banking_files = [
            p
            for p in art_root.rglob("*.json")
            if json.loads(p.read_text("utf-8"))["scenario_id"] == REAL_BANKING_ID
        ]
        assert banking_files
        seen_parse_fail = False
        for p in banking_files:
            payload = json.loads(p.read_text("utf-8"))
            for jr in payload["judges"]["task_completion"]:
                if jr["judge_name"] == "Claude Opus 4.6" and jr["parse_failed"]:
                    seen_parse_fail = True
        assert seen_parse_fail, "opus parse-failure not recorded in banking artifacts"

    # --- parquet row surface -------------------------------------------------

    def test_parquet_rows_complete(self):
        df = self.df
        # 4 scenarios * 2 models * 2 runs = 16 rows.
        assert len(df) == 16
        required = {
            "scenario_id",
            "domain",
            "category",
            "model",
            "holdout",
            "efficacy",
            "task_completion",
            "tool_selection",
            "state_score",
            "state_checks_passed",
            "state_checks_total",
            "cost_usd",
            "latency_ms",
            "total_turns",
            "completed",
            "ended_by",
            "state_progress_at_end",
            "premature_end",
            "failure_mode",
            "failure_mode_source",
            "tc_n_judges",
            "ts_n_judges",
            "tc_parse_failures",
            "ts_parse_failures",
            "reliability_pass_rate",
            "reliability_consistency",
            "reliability_pass_hat_1",
            "reliability_pass_hat_2",
        }
        assert required <= set(df.columns), required - set(df.columns)
        # State grading produced a NON-trivial partial score on banking (4/5=0.8).
        banking = df[df["scenario_id"] == REAL_BANKING_ID]
        assert (banking["state_checks_total"] == 5).all()
        assert (banking["state_checks_passed"] == 4).all()
        assert abs(banking["state_score"].iloc[0] - 0.8) < 1e-6

    def test_premature_and_normal_endings_both_present(self):
        df = self.df
        # Banking ends prematurely (state 0.8 < 1.0 when user sim says done).
        banking = df[df["scenario_id"] == REAL_BANKING_ID]
        assert banking["premature_end"].all()
        assert (banking["ended_by"] == "user_sim").all()
        # Holdout dummies: agent does nothing, world unchanged -> state 1.0, so the
        # user-sim end is NOT premature (a clean, goals-verifiably-met ending).
        hold = df[df["holdout"]]
        assert not hold["premature_end"].any()
        assert (hold["ended_by"] == "user_sim").all()

    def test_judge_parse_failure_accounting_in_rows(self):
        df = self.df
        banking = df[df["scenario_id"] == REAL_BANKING_ID]
        # Opus parse-failed on banking -> 2 valid judges, 1 parse failure per row.
        assert (banking["tc_n_judges"] == 2).all()
        assert (banking["tc_parse_failures"] == 1).all()
        # A clean scenario keeps all 3 judges.
        cs = df[df["scenario_id"] == REAL_CS_ID]
        assert (cs["tc_n_judges"] == 3).all()
        assert (cs["tc_parse_failures"] == 0).all()

    # --- manifest + pre-registration linkage ---------------------------------

    def test_manifest_links_pre_registration(self):
        manifest = json.loads((self.ctx.results_dir / "run_manifest.json").read_text("utf-8"))
        assert manifest["models_failed"] == []
        assert set(manifest["models_completed"]) == {"GPT-5.5", "Claude Sonnet 4.6"}
        assert manifest["scenario_counts"] == {"banking": 1, "customer_success": 1}
        assert manifest["holdout"]["n_scenarios"] == 2
        assert len(manifest["pre_registration"]["sha256"]) == 64
        assert manifest["pre_registration"]["file"] == PRE_REGISTRATION_FILENAME

    # --- leaderboard surface -------------------------------------------------

    def test_leaderboard_has_full_new_surface(self):
        lb = self.leaderboard
        assert {"GPT-5.5", "Claude Sonnet 4.6"} == {m["name"] for m in lb["models"]}
        # Top-level new surfaces.
        assert "judge_alpha" in lb
        assert lb["judge_alpha"]["task_completion"] is not None
        assert "length_bias" in lb
        assert lb["holdout"]["present"] is True
        assert lb["holdout"]["models_with_gap"] >= 1
        # Per-model new surfaces.
        for m in lb["models"]:
            assert m["reliability_pass_hat_k"]  # pass^k dict, non-empty
            assert "1" in m["reliability_pass_hat_k"]
            assert m["judge_deltas"] is not None
            assert m["premature_end_rate"] is not None
            assert m["holdout_score"] is not None
            assert m["holdout_gap"] is not None
            assert "rank_band" in m

    def test_leaderboard_excludes_holdout_scenario_detail(self):
        # Public efficacy on the board must NOT include holdout rows, and no
        # holdout scenario id may appear anywhere in the published JSON.
        flat = json.dumps(self.leaderboard)
        assert "9999" not in flat
        assert "e2e0" not in flat

    def test_public_efficacy_pinned_to_public_rows_only(self):
        # Behavioral leak check (review SHOULD-FIX on PR #52): the published
        # per-model efficacy must equal the mean over PUBLIC parquet rows
        # exactly — if holdout rows ever leak into the public aggregate, the
        # numbers diverge (holdout dummies score state 1.0 vs banking's 0.8,
        # so a leak moves the mean by construction).
        df = self.df
        public = df[~df["holdout"]]
        for m in self.leaderboard["models"]:
            expected = public[public["model"] == m["name"]]["efficacy"].mean()
            leaked = df[df["model"] == m["name"]]["efficacy"].mean()
            # The board publishes efficacy rounded to 4 decimals, so the
            # tripwire must stay sharp AT THAT PRECISION: a leak that rounding
            # would hide is a leak this test cannot see.
            assert round(expected, 4) != round(leaked, 4), (
                "test setup lost its tripwire: public-only and public+holdout "
                "means must differ at published precision for this check to "
                "mean anything"
            )
            assert m["efficacy"] == round(expected, 4)

    def test_corpus_health_and_consistency_bands_published(self):
        # Issue #71 end to end. The PUBLIC corpus here is 2 scenarios whose runs
        # ALL fail (efficacy < 0.7 — see the failure-modes test below), while the
        # 2 holdout dummies pass with state 1.0. So the corpus-health counts are
        # only correct if the holdout exclusion held: a leak moves total to 4
        # and the pass counts off zero by construction.
        health = self.leaderboard["corpus_health"]
        assert health["total_scenarios"] == 2
        assert health["passed_at_least_once"] == 0
        assert health["never_passed"] == 2
        assert health["passed_by_every_model"] == 0
        assert health["n_models"] == 2
        assert health["headline"] == (
            "0 of 2 scenarios passed at least once; 0 passed by every model"
        )
        # Consistency bands (solid/avg/best): present (2 reliability runs) and
        # pinned to the all-fail public rows — holdout passes would lift them.
        for m in self.leaderboard["models"]:
            band = m["consistency_band"]
            assert band["n_runs"] == 2
            assert band["n_scenarios"] == 2
            assert band["solid_rate"] == 0.0
            assert band["avg_pass_rate"] == 0.0
            assert band["best_of_rate"] == 0.0

    def test_failure_modes_and_macro_published(self):
        # Failure taxonomy (#55), end to end. Every PUBLIC run here fails (the
        # judge fakes + partial state land efficacy below 0.7) with the user sim
        # quitting before the state check passed -> the deterministic premature
        # flag classifies them all, ahead of any keyword matching. The holdout
        # dummies PASS (state 1.0) -> no failure mode.
        df = self.df
        public = df[~df["holdout"]]
        assert (public["failure_mode"] == "premature-end").all()
        assert (public["failure_mode_source"] == "premature-flag").all()
        assert df[df["holdout"]]["failure_mode"].isna().all()

        # Published profiles count PUBLIC rows only (2 scenarios * 2 runs = 4),
        # pinning the holdout exclusion the same way the micro efficacy is.
        for m in self.leaderboard["models"]:
            profile = m["failure_profile"]
            assert profile["n_rows"] == 4
            assert profile["n_failures"] == 4
            assert profile["modes"]["premature-end"]["count"] == 4
            # Macro efficacy published with a CI alongside the micro headline.
            assert m["efficacy_macro_category"] is not None
            assert m["efficacy_macro_domain"] is not None
            lo, hi = m["efficacy_macro_category_ci"]
            assert lo is not None and hi is not None
        assert self.leaderboard["failure_taxonomy"]["modes"]
        assert self.leaderboard["categories"] == ["adaptive_tool_use"]

    def test_judge_deltas_cover_every_judge(self):
        lb = self.leaderboard
        for m in lb["models"]:
            tc_deltas = m["judge_deltas"]["task_completion"]
            # All three judges represented in the per-judge columns (even opus,
            # which only parse-failed on banking, still scores cs).
            assert {"Kimi K2.6", "GLM-4.6", "Claude Opus 4.6"} <= set(tc_deltas)

    # --- publish gate --------------------------------------------------------

    def test_publish_gate_blocks_below_scenario_minimum(self):
        # This run has 1 public scenario per domain — below MIN_SCENARIOS_FOR_PUBLISH.
        # The gate must refuse (exit 1) on the REAL manifest this run wrote.
        manifest_path = self.ctx.results_dir / "run_manifest.json"
        assert check_publish_ready(manifest_path, allow_partial=False) == 1
        # ...but --allow-partial downgrades to a warning and exits 0.
        assert check_publish_ready(manifest_path, allow_partial=True) == 0


class TestPublishGateModelsFailed:
    """Publish gate on a complete-but-models-failed manifest vs a clean one.

    Built as small standalone manifests (the gate reads only the manifest), so we
    isolate the failed-models branch from the scenario-minimum branch."""

    def _manifest(self, tmp_path, *, failed, counts):
        m = {
            "models_failed": failed,
            "models_completed": ["GPT-5.5"],
            "scenario_counts": counts,
        }
        p = tmp_path / "run_manifest.json"
        p.write_text(json.dumps(m), encoding="utf-8")
        return p

    def test_clean_manifest_passes(self, tmp_path):
        # No failures, all domains at/above the minimum -> exit 0.
        p = self._manifest(tmp_path, failed=[], counts={"banking": 40, "customer_success": 40})
        assert check_publish_ready(p, allow_partial=False) == 0

    def test_models_failed_blocks(self, tmp_path):
        p = self._manifest(
            tmp_path, failed=["Gemini 3.1 Pro"], counts={"banking": 40, "customer_success": 40}
        )
        assert check_publish_ready(p, allow_partial=False) == 1
        assert check_publish_ready(p, allow_partial=True) == 0

    def test_missing_manifest_blocks(self, tmp_path):
        assert check_publish_ready(tmp_path / "nope.json", allow_partial=False) == 1


class TestNullAgentEndToEnd:
    """Run the null agent through the SAME e2e path (--include-null-agent).

    Pins the anti-gaming contract end to end: the do-nothing agent scores ~0 on
    the deterministic state checks AND is excluded from the published leaderboard.
    """

    def test_null_agent_scores_zero_state_and_is_excluded(self, offline_pipeline):
        ctx = offline_pipeline
        ctx.run_main(
            [
                "--models",
                "GPT-5.5",
                "--include-null-agent",
            ]
        )
        leaderboard, df = ctx.aggregate()

        # The null agent IS in the raw parquet (it ran)...
        assert NULL_AGENT_NAME in set(df["model"])
        null_rows = df[df["model"] == NULL_AGENT_NAME]
        # ...and it made no tool calls, so the banking state assertions all fail
        # (0/5) -> state score 0.0 on the state-graded scenarios.
        banking_null = null_rows[null_rows["scenario_id"] == REAL_BANKING_ID]
        assert (banking_null["state_score"] == 0.0).all()
        assert (banking_null["state_checks_passed"] == 0).all()

        # ...but it is EXCLUDED from the published leaderboard.
        board_names = {m["name"] for m in leaderboard["models"]}
        assert NULL_AGENT_NAME not in board_names
        assert "GPT-5.5" in board_names


class TestMaxTurnsEnding:
    """A scenario the user sim never ends -> ended_by == 'max_turns'."""

    def test_max_turns_ending(self, offline_pipeline, monkeypatch):
        ctx = offline_pipeline
        # Make the CS scenario run to the turn budget instead of completing.
        import sys as _sys

        monkeypatch.setattr(_sys.modules[__name__], "_RUN_TO_MAX_TURNS", {REAL_CS_ID})
        try:
            ctx.run_main(["--models", "GPT-5.5"])
            _, df = ctx.aggregate()
            cs = df[df["scenario_id"] == REAL_CS_ID]
            assert (cs["ended_by"] == "max_turns").all()
            assert not cs["completed"].any()
            # max-turns runs still record where the state check landed.
            assert cs["state_progress_at_end"].notna().all()
        finally:
            _RUN_TO_MAX_TURNS.clear()


class TestEnvironmentCaptureDegradation:
    """Capture failure must never lose the manifest of a completed run (H3)."""

    def test_capture_failure_degrades_to_marker(self, offline_pipeline, monkeypatch):
        ctx = offline_pipeline

        def boom(path):
            raise OSError("disk full")

        monkeypatch.setattr(run_eval, "capture_environment", boom)
        ctx.run_main(["--models", "GPT-5.5"])

        manifest = json.loads((ctx.results_dir / "run_manifest.json").read_text("utf-8"))
        # The manifest survives with an honest marker instead of an env block.
        assert manifest["environment"] == {"capture_failed": "disk full"}
        assert manifest["models_completed"] == ["GPT-5.5"]
