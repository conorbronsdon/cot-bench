"""Tests for dual control — user-side world actions (issue #58).

Pins the guarantees of the dual-control mechanism, all fully OFFLINE (stubbed
sims, no API calls, deterministic):

1. **Parsing + validation** — user_tools / user_actions / trigger bounds and the
   AUTHORIZATION SCOPE are enforced at both the object boundary (DualControl) and
   the on-disk validator, with the same rules (one definition of a valid block).
2. **Deterministic user actions** — a declared action fires at its trigger
   (after_turn / agent_called), its state delta is applied through the SAME
   apply_state_delta the agent's tools use, and its user_message is injected
   verbatim as the user turn; no dual_control means the loop is byte-identical.
3. **Attribution** — agent-side and user-side mutations are tracked separately;
   the coordination verdict is True iff the correct end state is reached AND the
   agent never wrote a user-owned path (no double-apply).
4. **Hash handling (the #54 lesson)** — dual_control is hashed scenario content
   when present, added CONDITIONALLY so single-control scenarios keep their exact
   prior digests; a dual_control change moves the digest.
5. **Aggregation** — a per-model coordination_rate over dual-control rows ONLY,
   emitted into leaderboard.json ONLY when such rows exist (conditional
   emission), never moving public efficacy.
6. **Demo fixtures** — the two demonstration fixtures (approve-mid-flow,
   act-first-no-double-apply) live OUTSIDE data/scenarios and validate clean.
"""

import json
from pathlib import Path

import pandas as pd
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from eval.config import DEFAULT_SIMULATION, Domain
from eval.pre_registration import _scenario_to_canonical_dict, scenario_set_hash
from eval.providers.registry import ModelSpec
from eval.simulation.dual_control import (
    TRIGGER_AFTER_TURN,
    TRIGGER_AGENT_CALLED,
    TRIGGER_KINDS,
    DualControl,
    UserAction,
    UserTool,
    action_fires,
)
from eval.simulation.runner import (
    CONVERSATION_COMPLETE,
    Scenario,
    SimulationRunner,
)
from scripts.aggregate_results import compute_dual_control_rates, compute_leaderboard
from scripts.validate_scenarios import validate_scenario_dict

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "dual_control"


# --------------------------------------------------------------------------- #
# Helpers: a banking scenario with a transfer task + an optional dual_control.
# --------------------------------------------------------------------------- #
def _ground_truth():
    return {
        "accounts": {
            "BUS-CHK-001": {"type": "checking", "balance": 1000.0},
        },
        "pending_requests": [
            {"request_id": "REQ-1", "amount": 500.0, "status": "awaiting_approval"},
        ],
    }


def _expected_changes():
    return [
        {
            "assert": "pending_requests",
            "op": "contains",
            "match": {"request_id": "REQ-1", "status": "approved"},
        },
    ]


def _approval_dc(
    trigger=TRIGGER_AGENT_CALLED, trigger_value="request_approval", message="Approved."
):
    return DualControl(
        user_tools=[
            UserTool(name="approve_request", scope=["pending_requests"]),
        ],
        user_actions=[
            UserAction(
                tool="approve_request",
                trigger=trigger,
                trigger_value=trigger_value,
                state_delta={
                    "pending_requests": [
                        {"request_id": "REQ-1", "amount": 500.0, "status": "approved"}
                    ]
                },
                user_message=message,
            )
        ],
    )


def _scenario(dual_control: DualControl | None = None, **overrides):
    kwargs = dict(
        id="banking_dc_0001",
        domain=Domain.BANKING,
        persona={"name": "Margaret"},
        user_goals=["approve a request", "check balance", "confirm done"],
        tools=[
            {"name": "request_approval", "description": "ask user to approve", "parameters": []},
            {"name": "execute", "description": "execute approved request", "parameters": []},
        ],
        category="adaptive_tool_use",
        initial_message="I need to approve something.",
        ground_truth=_ground_truth(),
        expected_state_changes=_expected_changes(),
        dual_control=dual_control,
    )
    kwargs.update(overrides)
    return Scenario(**kwargs)


class ScriptedAgent(BaseChatModel):
    """Fake agent returning one scripted text reply (no tools, no API)."""

    text: str = "Done."

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.text))])

    def bind_tools(self, tools, **kwargs):  # noqa: ARG002
        return self

    @property
    def _llm_type(self) -> str:
        return "scripted-agent"


class RecordingUserSim(BaseChatModel):
    """User sim that records the turns it is asked to generate.

    Returns CONVERSATION_COMPLETE after ``done_after`` generated turns so the run
    terminates. A fired user action with a user_message must NOT invoke this sim —
    a test asserts the recorded count to prove the action replaced a generated
    turn.
    """

    generated: int = 0
    done_after: int = 99

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.generated += 1
        text = CONVERSATION_COMPLETE if self.generated >= self.done_after else "keep going"
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    @property
    def _llm_type(self) -> str:
        return "recording-user-sim"


def _runner_with(user_sim, tool_sim_text="{}"):
    runner = SimulationRunner.__new__(SimulationRunner)
    runner.config = DEFAULT_SIMULATION
    runner._user_sim = user_sim

    class _ConstSim(BaseChatModel):
        text: str = tool_sim_text

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.text))])

        @property
        def _llm_type(self):
            return "const-sim"

    runner._tool_sim = _ConstSim()
    return runner


SPEC = ModelSpec(name="FakeModel", model_id="fake", provider="openai")


# --------------------------------------------------------------------------- #
# 1. Parsing + validation (including the authorization boundary)
# --------------------------------------------------------------------------- #
class TestParsing:
    def test_trigger_enum(self):
        assert TRIGGER_KINDS == {TRIGGER_AFTER_TURN, TRIGGER_AGENT_CALLED}

    def test_from_dict_none_is_none(self):
        assert DualControl.from_dict(None) is None
        assert DualControl.from_dict({}) is None

    def test_valid_block_parses(self):
        dc = DualControl.from_dict(
            {
                "user_tools": [{"name": "approve_request", "scope": ["pending_requests"]}],
                "user_actions": [
                    {
                        "tool": "approve_request",
                        "trigger": "agent_called",
                        "trigger_value": "request_approval",
                        "state_delta": {"pending_requests": []},
                    }
                ],
            }
        )
        assert list(dc.user_tools) == ["approve_request"]
        assert dc.user_actions[0].trigger == TRIGGER_AGENT_CALLED

    def test_no_user_tools_raises(self):
        with pytest.raises(ValueError, match="at least one user_tool"):
            DualControl.from_dict({"user_tools": [], "user_actions": [{"tool": "x"}]})

    def test_no_user_actions_raises(self):
        with pytest.raises(ValueError, match="at least one user_action"):
            DualControl.from_dict({"user_tools": [{"name": "t", "scope": []}], "user_actions": []})

    def test_action_referencing_undeclared_tool_raises(self):
        with pytest.raises(ValueError, match="undeclared user_tool"):
            DualControl(
                user_tools=[UserTool(name="approve_request", scope=["pending_requests"])],
                user_actions=[UserAction(tool="nope", trigger=TRIGGER_AFTER_TURN, trigger_value=2)],
            )

    def test_action_out_of_scope_raises(self):
        # The authorization boundary: an action that writes a key the tool's
        # declared scope does NOT cover is rejected at construction.
        with pytest.raises(ValueError, match="outside its declared scope"):
            DualControl(
                user_tools=[UserTool(name="approve_request", scope=["pending_requests"])],
                user_actions=[
                    UserAction(
                        tool="approve_request",
                        trigger=TRIGGER_AFTER_TURN,
                        trigger_value=2,
                        state_delta={"accounts.BUS-CHK-001.balance": 0.0},
                    )
                ],
            )

    @pytest.mark.parametrize("turn", [0, 10, 99])
    def test_after_turn_out_of_range_raises(self, turn):
        with pytest.raises(ValueError, match="out of range"):
            UserAction(tool="t", trigger=TRIGGER_AFTER_TURN, trigger_value=turn)

    def test_agent_called_requires_string(self):
        with pytest.raises(ValueError, match="non-empty tool name"):
            UserAction(tool="t", trigger=TRIGGER_AGENT_CALLED, trigger_value=5)

    def test_unknown_trigger_raises(self):
        with pytest.raises(ValueError, match="Unknown user_action trigger"):
            UserAction(tool="t", trigger="someday", trigger_value=2)


class TestActionFires:
    def test_after_turn_fires_at_turn(self):
        a = UserAction(tool="t", trigger=TRIGGER_AFTER_TURN, trigger_value=3)
        assert action_fires(a, next_user_turn=2, agent_tool_calls_so_far=set()) is False
        assert action_fires(a, next_user_turn=3, agent_tool_calls_so_far=set()) is True

    def test_agent_called_fires_after_call(self):
        a = UserAction(tool="t", trigger=TRIGGER_AGENT_CALLED, trigger_value="req")
        assert action_fires(a, next_user_turn=2, agent_tool_calls_so_far=set()) is False
        assert action_fires(a, next_user_turn=2, agent_tool_calls_so_far={"req"}) is True


# --------------------------------------------------------------------------- #
# 2. Deterministic user actions in the runner
# --------------------------------------------------------------------------- #
class TestUserActionInRunner:
    def test_after_turn_action_injects_message_and_mutates(self, monkeypatch):
        # User action at turn 2 with a user_message must appear verbatim as the
        # turn-2 user message, the user sim must NOT have generated it, and the
        # shared world must carry the user's state delta.
        dc = DualControl(
            user_tools=[UserTool(name="approve_request", scope=["pending_requests"])],
            user_actions=[
                UserAction(
                    tool="approve_request",
                    trigger=TRIGGER_AFTER_TURN,
                    trigger_value=2,
                    state_delta={
                        "pending_requests": [
                            {"request_id": "REQ-1", "amount": 500.0, "status": "approved"}
                        ]
                    },
                    user_message="I approved it myself.",
                )
            ],
        )
        sim = RecordingUserSim(generated=0, done_after=99)
        runner = _runner_with(sim)
        monkeypatch.setattr("eval.simulation.runner.create_model", lambda spec: ScriptedAgent())
        result = runner.run(_scenario(dc), SPEC)

        user_turns = [t.content for t in result.turns if t.role == "user"]
        assert user_turns[2] == "I approved it myself."
        assert result.final_world["pending_requests"][0]["status"] == "approved"
        assert result.user_actions_fired == 1
        # The action turn replaced a generated one (10 turns - 1 injected).
        assert sim.generated == 9

    def test_no_dual_control_does_not_inject(self, monkeypatch):
        sim = RecordingUserSim(generated=0, done_after=99)
        runner = _runner_with(sim)
        monkeypatch.setattr("eval.simulation.runner.create_model", lambda spec: ScriptedAgent())
        result = runner.run(_scenario(None), SPEC)
        assert all("approved" not in t.content for t in result.turns)
        assert result.dual_control is False
        assert result.user_actions_fired == 0
        assert result.coordination_ok is None
        assert sim.generated == 10

    def test_action_deterministic_across_runs(self, monkeypatch):
        dc = _approval_dc(message="Approved on my end.")
        monkeypatch.setattr("eval.simulation.runner.create_model", lambda spec: ScriptedAgent())
        seen = []
        for _ in range(3):
            sim = RecordingUserSim(generated=0, done_after=99)
            # agent_called trigger needs the agent to call request_approval; our
            # ScriptedAgent calls no tools, so this action never fires — assert
            # the (deterministic) non-firing is itself identical run to run.
            res = _runner_with(sim).run(_scenario(dc), SPEC)
            seen.append(res.user_actions_fired)
        assert seen == [0, 0, 0]


# --------------------------------------------------------------------------- #
# 3. Attribution + coordination verdict
# --------------------------------------------------------------------------- #
class TestCoordinationVerdict:
    def test_none_when_not_dual_control(self):
        assert (
            SimulationRunner._coordination_verdict(_scenario(None), _ground_truth(), set(), set())
            is None
        )

    def test_none_when_no_ground_truth(self):
        dc = _approval_dc()
        scen = _scenario(dc, ground_truth=None, expected_state_changes=None)
        assert SimulationRunner._coordination_verdict(scen, None, set(), set()) is None

    def test_recovered_when_end_state_and_no_trespass(self):
        dc = _approval_dc()
        final = {
            "accounts": {"BUS-CHK-001": {"type": "checking", "balance": 1000.0}},
            "pending_requests": [{"request_id": "REQ-1", "amount": 500.0, "status": "approved"}],
        }
        # Agent mutated only an agent-owned key; user owned pending_requests.
        verdict = SimulationRunner._coordination_verdict(
            _scenario(dc),
            final,
            agent_mutated_keys={"wires_sent"},
            user_mutated_keys={"pending_requests"},
        )
        assert verdict is True

    def test_not_coordinated_when_agent_double_applies(self):
        # The agent wrote the user-owned 'pending_requests' key itself — the
        # canonical double-apply. Even though the end state matches, attribution
        # fails, so coordination is False.
        dc = _approval_dc()
        final = {
            "accounts": {"BUS-CHK-001": {"type": "checking", "balance": 1000.0}},
            "pending_requests": [{"request_id": "REQ-1", "amount": 500.0, "status": "approved"}],
        }
        verdict = SimulationRunner._coordination_verdict(
            _scenario(dc),
            final,
            agent_mutated_keys={"pending_requests"},
            user_mutated_keys={"pending_requests"},
        )
        assert verdict is False

    def test_not_coordinated_when_end_state_wrong(self):
        dc = _approval_dc()
        final = _ground_truth()  # nothing approved
        verdict = SimulationRunner._coordination_verdict(
            _scenario(dc), final, agent_mutated_keys=set(), user_mutated_keys=set()
        )
        assert verdict is False

    def test_run_stamps_dual_control_and_verdict(self, monkeypatch):
        # after_turn action fires -> coordination graded (not None).
        dc = _approval_dc(trigger=TRIGGER_AFTER_TURN, trigger_value=2)
        sim = RecordingUserSim(generated=0, done_after=99)
        monkeypatch.setattr("eval.simulation.runner.create_model", lambda spec: ScriptedAgent())
        result = _runner_with(sim).run(_scenario(dc), SPEC)
        assert result.dual_control is True
        assert result.user_actions_fired == 1
        assert result.coordination_ok in (True, False)

    def test_no_action_fired_yields_none_verdict(self, monkeypatch):
        # agent_called trigger that never fires (ScriptedAgent calls no tools):
        # coordination_ok must be None (no coordination to grade), not False.
        dc = _approval_dc(trigger=TRIGGER_AGENT_CALLED, trigger_value="request_approval")
        sim = RecordingUserSim(generated=0, done_after=99)
        monkeypatch.setattr("eval.simulation.runner.create_model", lambda spec: ScriptedAgent())
        result = _runner_with(sim).run(_scenario(dc), SPEC)
        assert result.dual_control is True
        assert result.user_actions_fired == 0
        assert result.coordination_ok is None


# --------------------------------------------------------------------------- #
# 4. Hash handling — the #54 lesson
# --------------------------------------------------------------------------- #
class TestHashHandling:
    def test_single_control_digest_unchanged(self):
        data = _scenario_to_canonical_dict(_scenario(None))
        assert "dual_control" not in data

    def test_dual_control_appears_in_canonical_dict(self):
        dc = _approval_dc()
        data = _scenario_to_canonical_dict(_scenario(dc))
        assert data["dual_control"]["user_tools"][0]["name"] == "approve_request"
        assert data["dual_control"]["user_actions"][0]["trigger"] == TRIGGER_AGENT_CALLED

    def test_dual_control_changes_corpus_hash(self):
        without = {Domain.BANKING: [_scenario(None)]}
        with_dc = {Domain.BANKING: [_scenario(_approval_dc())]}
        assert scenario_set_hash(without)[0] != scenario_set_hash(with_dc)[0]

    def test_different_action_changes_hash(self):
        h1 = scenario_set_hash({Domain.BANKING: [_scenario(_approval_dc(message="A"))]})[0]
        h2 = scenario_set_hash({Domain.BANKING: [_scenario(_approval_dc(message="B"))]})[0]
        assert h1 != h2

    def test_single_control_digest_matches_legacy_object(self):
        from types import SimpleNamespace

        legacy = SimpleNamespace(
            id="banking_dc_0001",
            domain=Domain.BANKING,
            persona={"name": "Margaret"},
            user_goals=["approve a request", "check balance", "confirm done"],
            tools=[
                {
                    "name": "request_approval",
                    "description": "ask user to approve",
                    "parameters": [],
                },
                {"name": "execute", "description": "execute approved request", "parameters": []},
            ],
            category="adaptive_tool_use",
            initial_message="I need to approve something.",
            ground_truth=_ground_truth(),
            expected_state_changes=_expected_changes(),
        )
        assert _scenario_to_canonical_dict(legacy) == _scenario_to_canonical_dict(_scenario(None))


# --------------------------------------------------------------------------- #
# 5. Validator
# --------------------------------------------------------------------------- #
class TestValidator:
    def _base(self):
        return {
            "id": "banking_x_0001",
            "category": "adaptive_tool_use",
            "schema_version": "0.2",
            "authorship": {"author_model": "human-handwritten"},
            "persona": {
                "name": "M",
                "age": 50,
                "occupation": "x",
                "personality_traits": ["a"],
                "tone": "t",
                "detail_level": "moderate",
                "background": "b",
            },
            "user_goals": ["g1", "g2", "g3"],
            "tools": [
                {"name": "request_approval", "description": "d", "parameters": []},
                {"name": "execute", "description": "d", "parameters": []},
            ],
            "initial_message": "hello there I need help",
            "ground_truth": {
                "pending_requests": [{"request_id": "REQ-1", "status": "awaiting_approval"}]
            },
            "expected_state_changes": [],
        }

    def _valid_dc(self):
        return {
            "user_tools": [{"name": "approve_request", "scope": ["pending_requests"]}],
            "user_actions": [
                {
                    "tool": "approve_request",
                    "trigger": "agent_called",
                    "trigger_value": "request_approval",
                    "state_delta": {"pending_requests": []},
                }
            ],
        }

    def test_valid_block_passes(self):
        data = self._base()
        data["dual_control"] = self._valid_dc()
        assert validate_scenario_dict(data) == []

    def test_no_block_unaffected(self):
        assert validate_scenario_dict(self._base()) == []

    def test_undeclared_tool_rejected(self):
        data = self._base()
        dc = self._valid_dc()
        dc["user_actions"][0]["tool"] = "ghost"
        data["dual_control"] = dc
        errs = validate_scenario_dict(data)
        assert any("undeclared user_tool" in e for e in errs)

    def test_out_of_scope_delta_rejected(self):
        data = self._base()
        dc = self._valid_dc()
        dc["user_actions"][0]["state_delta"] = {"accounts.X.balance": 0}
        data["dual_control"] = dc
        errs = validate_scenario_dict(data)
        assert any("outside the declared scope" in e for e in errs)

    def test_bad_trigger_rejected(self):
        data = self._base()
        dc = self._valid_dc()
        dc["user_actions"][0]["trigger"] = "whenever"
        data["dual_control"] = dc
        errs = validate_scenario_dict(data)
        assert any("unknown trigger" in e for e in errs)

    def test_bad_after_turn_rejected(self):
        data = self._base()
        dc = self._valid_dc()
        dc["user_actions"][0]["trigger"] = "after_turn"
        dc["user_actions"][0]["trigger_value"] = 99
        data["dual_control"] = dc
        errs = validate_scenario_dict(data)
        assert any("out of range" in e for e in errs)

    def test_empty_tools_rejected(self):
        data = self._base()
        dc = self._valid_dc()
        dc["user_tools"] = []
        data["dual_control"] = dc
        errs = validate_scenario_dict(data)
        assert any("non-empty 'user_tools'" in e for e in errs)


# --------------------------------------------------------------------------- #
# 6. Aggregation: per-model coordination_rate, conditional emission
# --------------------------------------------------------------------------- #
def _dc_df(verdicts_by_model, n_runs=2):
    rows = []
    for model, verdicts in verdicts_by_model.items():
        for i, ok in enumerate(verdicts):
            for r in range(n_runs):
                rows.append(
                    {
                        "scenario_id": f"dc_scen_{i:02d}",
                        "domain": "banking",
                        "category": "adaptive_tool_use",
                        "model": model,
                        "holdout": False,
                        "sim_profile": "cooperative",
                        "efficacy": 0.9 if ok else 0.2,
                        "task_completion": 0.9,
                        "tool_selection": 0.9,
                        "cost_usd": 0.01,
                        "latency_ms": 2000.0,
                        "total_turns": 7,
                        "reliability_pass_rate": 0.9,
                        "reliability_consistency": 0.9,
                        "tc_agreement": 0.9,
                        "ts_agreement": 0.9,
                        "dual_control": True,
                        "user_actions_fired": 1,
                        "coordination_ok": ok,
                    }
                )
    return pd.DataFrame(rows)


def _plain_df(n_scen=3, n_runs=2):
    rows = []
    for s in range(n_scen):
        for r in range(n_runs):
            rows.append(
                {
                    "scenario_id": f"scen_{s:02d}",
                    "domain": "banking",
                    "category": "adaptive_tool_use",
                    "model": "A",
                    "holdout": False,
                    "sim_profile": "cooperative",
                    "efficacy": 0.8,
                    "task_completion": 0.8,
                    "tool_selection": 0.8,
                    "cost_usd": 0.01,
                    "latency_ms": 2000.0,
                    "total_turns": 5,
                    "reliability_pass_rate": 0.8,
                    "reliability_consistency": 0.9,
                    "tc_agreement": 0.9,
                    "ts_agreement": 0.9,
                    "dual_control": False,
                    "user_actions_fired": 0,
                    "coordination_ok": None,
                }
            )
    return pd.DataFrame(rows)


class TestAggregation:
    def test_rate_over_dual_control_rows(self):
        df = _dc_df({"A": [True, True, False, True]})  # 3/4
        table = compute_dual_control_rates(df)
        assert table["A"]["coordination_rate"] == 0.75
        assert table["A"]["n_dual_control_scenarios"] == 4
        assert table["A"]["n_dual_control_rows"] == 8

    def test_unfired_rows_excluded(self):
        df = _dc_df({"A": [True, False]})  # rate 0.5 over 4 rows
        unfired = df.iloc[[0]].copy()
        unfired["coordination_ok"] = None
        unfired["user_actions_fired"] = 0
        table = compute_dual_control_rates(pd.concat([df, unfired], ignore_index=True))
        assert table["A"]["coordination_rate"] == 0.5
        assert table["A"]["n_dual_control_rows"] == 4

    def test_empty_when_no_dual_control_rows(self):
        assert compute_dual_control_rates(_plain_df()) == {}
        assert compute_dual_control_rates(pd.DataFrame()) == {}

    def test_missing_column_yields_empty(self):
        df = _plain_df().drop(columns=["coordination_ok"])
        assert compute_dual_control_rates(df) == {}

    def test_deterministic(self):
        df = _dc_df({"A": [True, False, True]})
        assert compute_dual_control_rates(df) == compute_dual_control_rates(df)


class TestEmission:
    def test_emitted_when_dual_control_rows_present(self):
        df = _dc_df({"A": [True, False, True, True]})
        lb = compute_leaderboard(df)
        assert "dual_control_robustness" in lb
        assert lb["dual_control_robustness"]["models"]["A"]["coordination_rate"] == 0.75

    def test_absent_on_normal_run(self):
        lb = compute_leaderboard(_plain_df())
        assert "dual_control_robustness" not in lb

    def test_dual_control_rows_additive_not_replacing(self):
        df = _dc_df({"A": [True, True, True, True]})
        lb = compute_leaderboard(df)
        assert "dual_control_robustness" in lb
        assert any(m["name"] == "A" for m in lb["models"])


# --------------------------------------------------------------------------- #
# 7. Demonstration fixtures validate clean and carry the right shape
# --------------------------------------------------------------------------- #
class TestDemoFixtures:
    def test_fixtures_exist_outside_data_scenarios(self):
        files = sorted(FIXTURE_DIR.glob("*.json"))
        assert len(files) == 2
        for f in files:
            assert "data" not in f.parts or "scenarios" not in f.parts

    @pytest.mark.parametrize(
        "filename,trigger",
        [
            ("banking_user_approves_midflow.json", TRIGGER_AGENT_CALLED),
            ("cs_user_acts_first_no_double_apply.json", TRIGGER_AFTER_TURN),
        ],
    )
    def test_fixture_validates_and_parses(self, filename, trigger):
        data = json.loads((FIXTURE_DIR / filename).read_text(encoding="utf-8"))
        errors = validate_scenario_dict(data)
        assert errors == [], f"{filename}: {errors}"
        dc = DualControl.from_dict(data["dual_control"])
        assert dc.user_actions[0].trigger == trigger

    def test_fixtures_cover_both_coordination_shapes(self):
        triggers = set()
        for f in FIXTURE_DIR.glob("*.json"):
            data = json.loads(f.read_text(encoding="utf-8"))
            dc = DualControl.from_dict(data["dual_control"])
            triggers.add(dc.user_actions[0].trigger)
        # One approve-mid-flow (agent_called) + one act-first (after_turn).
        assert triggers == {TRIGGER_AGENT_CALLED, TRIGGER_AFTER_TURN}
