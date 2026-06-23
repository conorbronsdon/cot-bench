"""Tests for recovery probes (issue #57).

Pins the guarantees of the recovery-probe mechanism, all fully OFFLINE
(stubbed sims, no API calls, deterministic):

1. **Probe parsing + validation** — kind/turn/injection bounds enforced at both
   the object boundary (RecoveryProbe) and the on-disk validator, with the same
   rules (one definition of a valid probe).
2. **Deterministic scripted injection** — the probe text is injected verbatim at
   the declared turn, REPLACING the user-sim message, identical every run; no
   probe means the loop is byte-identical to before.
3. **Recovery scoring** — recovered iff the normal end state is reached AND the
   probe's recovery_assertions hold (the bad entity was not acted on); a partial
   counts as non-recovery; reuses the existing state grader (no new machinery).
4. **Hash handling (the #54 lesson)** — recovery_probe is hashed scenario
   content when present, added CONDITIONALLY so probe-less scenarios keep their
   exact prior digests; a probe change moves the digest.
5. **Aggregation** — a per-model recovery_rate over probe rows ONLY, emitted into
   leaderboard.json ONLY when probe rows exist (conditional emission), never
   moving public efficacy.
6. **Demo fixtures** — the three demonstration fixtures (one per probe kind)
   live OUTSIDE data/scenarios and validate clean.
"""

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from eval.config import DEFAULT_SIMULATION, Domain
from eval.pre_registration import _scenario_to_canonical_dict, scenario_set_hash
from eval.providers.registry import ModelSpec
from eval.simulation.probes import (
    CONTRADICTORY_REFERENCE,
    INCOMPLETE_ACTION_CLAIM,
    PROBE_KINDS,
    WRONG_ENTITY,
    RecoveryProbe,
)
from eval.simulation.runner import (
    CONVERSATION_COMPLETE,
    Scenario,
    SimulationRunner,
)
from eval.templating import instantiate
from scripts.aggregate_results import compute_leaderboard, compute_recovery_rates
from scripts.run_eval import _scenario_from_dict
from scripts.validate_scenarios import validate_scenario_dict

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "recovery_probes"


# --------------------------------------------------------------------------- #
# Helpers: a banking scenario with a transfer task + a wrong_entity probe.
# --------------------------------------------------------------------------- #
def _ground_truth():
    return {
        "accounts": {
            "BUS-CHK-001": {"type": "checking", "balance": 1000.0},
            "BUS-SAV-002": {"type": "savings", "balance": 5000.0},
        },
    }


def _expected_changes():
    return [
        {"assert": "accounts.BUS-CHK-001.balance", "op": "increased_by", "value": 500.0},
    ]


def _scenario(probe: RecoveryProbe | None = None, **overrides):
    kwargs = dict(
        id="banking_probe_0001",
        domain=Domain.BANKING,
        persona={"name": "Margaret"},
        user_goals=["transfer funds", "check balance", "report fraud"],
        tools=[
            {"name": "initiate_transfer", "description": "move money", "parameters": []},
        ],
        category="adaptive_tool_use",
        initial_message="I need to move some money.",
        ground_truth=_ground_truth(),
        expected_state_changes=_expected_changes(),
        recovery_probe=probe,
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
    """User sim that records the (turn) messages it is asked to generate.

    Returns CONVERSATION_COMPLETE after ``done_after`` generated turns so the run
    terminates. Crucially, the probe turn must NOT invoke this sim — the test
    asserts the recorded count to prove the probe replaced a generated turn.
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
# 1. Probe parsing + validation
# --------------------------------------------------------------------------- #
class TestProbeParsing:
    def test_enum_is_three_kinds(self):
        assert PROBE_KINDS == {CONTRADICTORY_REFERENCE, WRONG_ENTITY, INCOMPLETE_ACTION_CLAIM}

    def test_from_dict_none_is_none(self):
        assert RecoveryProbe.from_dict(None) is None
        assert RecoveryProbe.from_dict({}) is None

    def test_valid_probe_parses(self):
        p = RecoveryProbe.from_dict(
            {"turn": 4, "kind": WRONG_ENTITY, "injection": "send to BUS-CHK-999"}
        )
        assert p.turn == 4
        assert p.kind == WRONG_ENTITY
        assert p.injected_message() == "send to BUS-CHK-999"
        assert p.recovery_assertions == []  # normalized from absent

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="Unknown recovery-probe kind"):
            RecoveryProbe(turn=4, kind="surprise", injection="x")

    @pytest.mark.parametrize("turn", [0, 1, 3, 6, 9])
    def test_turn_out_of_range_raises(self, turn):
        with pytest.raises(ValueError, match="out of range"):
            RecoveryProbe(turn=turn, kind=WRONG_ENTITY, injection="x")

    def test_empty_injection_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            RecoveryProbe(turn=4, kind=WRONG_ENTITY, injection="   ")

    def test_bool_turn_rejected(self):
        # True is an int in Python; a probe turn must be a real int, not a bool.
        with pytest.raises(ValueError, match="must be an int"):
            RecoveryProbe(turn=True, kind=WRONG_ENTITY, injection="x")


# --------------------------------------------------------------------------- #
# 2. Deterministic scripted injection
# --------------------------------------------------------------------------- #
class TestInjection:
    def test_probe_injected_at_declared_turn(self, monkeypatch):
        # The probe at turn 4 must appear as the user message on turn 4 of the
        # transcript, verbatim, and the user sim must NOT have generated it.
        probe = RecoveryProbe(
            turn=4, kind=WRONG_ENTITY, injection="Send it to BUS-CHK-999 instead."
        )
        sim = RecordingUserSim(generated=0, done_after=99)  # never self-completes
        runner = _runner_with(sim)
        monkeypatch.setattr("eval.simulation.runner.create_model", lambda spec: ScriptedAgent())
        result = runner.run(_scenario(probe), SPEC)

        user_turns = [t.content for t in result.turns if t.role == "user"]
        # turn index 4 is the 5th user message (0-based loop).
        assert user_turns[4] == "Send it to BUS-CHK-999 instead."
        # The probe turn replaced a generated one. The runner generates one user
        # turn per outer iteration EXCEPT the initial message (turn 0 is the
        # scenario's initial_message, not sim-generated) and the probe turn. The
        # probe at turn 4 is injected at the END of iteration turn_num==3 (the
        # message for the NEXT turn), so iteration 3 skips its sim generation.
        # max_turns=10 => 10 generations normally, minus 1 for the skipped probe.
        assert sim.generated == 9

    def test_no_probe_does_not_inject(self, monkeypatch):
        sim = RecordingUserSim(generated=0, done_after=99)
        runner = _runner_with(sim)
        monkeypatch.setattr("eval.simulation.runner.create_model", lambda spec: ScriptedAgent())
        result = runner.run(_scenario(None), SPEC)
        # No probe text anywhere; the sim generated a next-turn message on every
        # one of the 10 outer iterations (turn 0's user message is the initial
        # message, but the sim still runs at the end of every iteration).
        assert all("BUS-CHK-999" not in t.content for t in result.turns)
        assert sim.generated == 10

    def test_injection_is_identical_across_runs(self, monkeypatch):
        probe = RecoveryProbe(turn=5, kind=CONTRADICTORY_REFERENCE, injection="Sam said 15%.")
        monkeypatch.setattr("eval.simulation.runner.create_model", lambda spec: ScriptedAgent())
        runs = []
        for _ in range(3):
            sim = RecordingUserSim(generated=0, done_after=99)
            res = _runner_with(sim).run(_scenario(probe), SPEC)
            runs.append([t.content for t in res.turns if t.role == "user"][5])
        assert runs == ["Sam said 15%."] * 3  # byte-identical, reproducible


# --------------------------------------------------------------------------- #
# 3. Recovery scoring (deterministic state grading)
# --------------------------------------------------------------------------- #
class TestRecoveryVerdict:
    def test_no_probe_yields_none(self):
        assert SimulationRunner._recovery_verdict(_scenario(None), _ground_truth()) is None

    def test_no_ground_truth_yields_none(self):
        probe = RecoveryProbe(turn=4, kind=WRONG_ENTITY, injection="x")
        scen = _scenario(probe, ground_truth=None, expected_state_changes=None)
        assert SimulationRunner._recovery_verdict(scen, None) is None

    def test_recovered_when_base_and_extra_pass(self):
        # Correct end state reached (checking +500) AND the wrong account never
        # appeared (recovery assertion: BUS-CHK-999 == null).
        probe = RecoveryProbe(
            turn=4,
            kind=WRONG_ENTITY,
            injection="send to BUS-CHK-999",
            recovery_assertions=[{"assert": "accounts.BUS-CHK-999", "op": "not_exists"}],
        )
        final = {
            "accounts": {
                "BUS-CHK-001": {"type": "checking", "balance": 1500.0},
                "BUS-SAV-002": {"type": "savings", "balance": 4500.0},
            }
        }
        assert SimulationRunner._recovery_verdict(_scenario(probe), final) is True

    def test_not_recovered_when_acted_on_wrong_entity(self):
        # Task done, but the agent CREATED the wrong account — recovery assertion
        # fails, so recovered is False even though the base change passed.
        probe = RecoveryProbe(
            turn=4,
            kind=WRONG_ENTITY,
            injection="send to BUS-CHK-999",
            recovery_assertions=[{"assert": "accounts.BUS-CHK-999", "op": "not_exists"}],
        )
        final = {
            "accounts": {
                "BUS-CHK-001": {"type": "checking", "balance": 1500.0},
                "BUS-SAV-002": {"type": "savings", "balance": 4500.0},
                "BUS-CHK-999": {"type": "checking", "balance": 500.0},
            }
        }
        assert SimulationRunner._recovery_verdict(_scenario(probe), final) is False

    def test_not_recovered_when_task_abandoned(self):
        # Wrong entity avoided, but the base task never completed (no +500).
        probe = RecoveryProbe(
            turn=4,
            kind=WRONG_ENTITY,
            injection="send to BUS-CHK-999",
            recovery_assertions=[{"assert": "accounts.BUS-CHK-999", "op": "not_exists"}],
        )
        final = _ground_truth()  # unchanged world
        assert SimulationRunner._recovery_verdict(_scenario(probe), final) is False

    def test_recovered_with_no_extra_assertions_uses_base_only(self):
        probe = RecoveryProbe(turn=4, kind=INCOMPLETE_ACTION_CLAIM, injection="already done?")
        final = {
            "accounts": {
                "BUS-CHK-001": {"type": "checking", "balance": 1500.0},
                "BUS-SAV-002": {"type": "savings", "balance": 4500.0},
            }
        }
        assert SimulationRunner._recovery_verdict(_scenario(probe), final) is True

    def test_run_stamps_kind_and_verdict(self, monkeypatch):
        probe = RecoveryProbe(turn=4, kind=WRONG_ENTITY, injection="send to BUS-CHK-999")
        sim = RecordingUserSim(generated=0, done_after=99)
        monkeypatch.setattr("eval.simulation.runner.create_model", lambda spec: ScriptedAgent())
        result = _runner_with(sim).run(_scenario(probe), SPEC)
        assert result.recovery_probe_kind == WRONG_ENTITY
        assert result.probe_fired is True  # turn 4 reached; the fault was delivered
        assert result.recovered in (True, False)  # graded, not None (has ground_truth)

    def test_non_probe_run_has_null_probe_fields(self, monkeypatch):
        sim = RecordingUserSim(generated=0, done_after=99)
        monkeypatch.setattr("eval.simulation.runner.create_model", lambda spec: ScriptedAgent())
        result = _runner_with(sim).run(_scenario(None), SPEC)
        assert result.recovery_probe_kind is None
        assert result.recovered is None
        assert result.probe_fired is False


# --------------------------------------------------------------------------- #
# 3b. Probe firing (review fix): recovered is None when the probe never fired
# --------------------------------------------------------------------------- #
class TestProbeFiring:
    """A declared probe that never FIRES must yield recovered=None.

    A probe declared at turn N only fires if the conversation actually reaches
    turn N. Before the fix, recovery was graded off probe DECLARATION, so a run
    that ended early was scored against a fault that was never injected — and
    when the base task happened to already be satisfied (here: empty
    expected_state_changes over an unchanged world) the row was falsely
    credited recovered=True. Both non-firing paths must stamp probe_fired=False
    and recovered=None, which compute_recovery_rates drops from recovery_rate.
    """

    INJECTION = "Send it to BUS-CHK-999 instead."

    def _probe(self, turn=4):
        return RecoveryProbe(turn=turn, kind=WRONG_ENTITY, injection=self.INJECTION)

    def test_early_completion_probe_not_fired(self, monkeypatch):
        # User sim emits CONVERSATION_COMPLETE on turn 1, probe declared at
        # turn 4. Empty expected_state_changes pass on the unchanged world —
        # the pre-fix false-credit case (recovered would have been True).
        scen = _scenario(self._probe(turn=4), expected_state_changes=[])
        sim = RecordingUserSim(generated=0, done_after=2)
        monkeypatch.setattr("eval.simulation.runner.create_model", lambda spec: ScriptedAgent())
        result = _runner_with(sim).run(scen, SPEC)

        assert result.ended_by == "user_sim"  # ended before the probe turn
        assert all(self.INJECTION not in t.content for t in result.turns)  # never delivered
        assert result.probe_fired is False
        assert result.recovered is None  # NOT True/False — no fault ever occurred
        assert result.recovery_probe_kind == WRONG_ENTITY  # declaration still auditable

    def test_max_turns_below_probe_turn_not_fired(self, monkeypatch):
        # max_turns=3 < probe.turn=5: the loop never reaches the staging
        # iteration, so the probe is never injected at all.
        scen = _scenario(self._probe(turn=5), expected_state_changes=[])
        sim = RecordingUserSim(generated=0, done_after=99)  # never self-completes
        runner = _runner_with(sim)
        runner.config = replace(DEFAULT_SIMULATION, max_turns=3)
        monkeypatch.setattr("eval.simulation.runner.create_model", lambda spec: ScriptedAgent())
        result = runner.run(scen, SPEC)

        assert result.ended_by == "max_turns"
        assert all(self.INJECTION not in t.content for t in result.turns)
        assert result.probe_fired is False
        assert result.recovered is None
        assert result.recovery_probe_kind == WRONG_ENTITY

    def test_probe_staged_on_final_iteration_not_fired(self, monkeypatch):
        # Edge between the two cases above: probe.turn == max_turns. The probe
        # text is STAGED as the next user message at the end of the loop's last
        # iteration, but the loop ends before it is ever delivered — the agent
        # never sees the fault, so it must not count as fired.
        scen = _scenario(self._probe(turn=4), expected_state_changes=[])
        sim = RecordingUserSim(generated=0, done_after=99)
        runner = _runner_with(sim)
        runner.config = replace(DEFAULT_SIMULATION, max_turns=4)
        monkeypatch.setattr("eval.simulation.runner.create_model", lambda spec: ScriptedAgent())
        result = runner.run(scen, SPEC)

        assert all(self.INJECTION not in t.content for t in result.turns)
        assert result.probe_fired is False
        assert result.recovered is None

    def test_fired_probe_still_grades(self, monkeypatch):
        # Control: when the conversation DOES reach the probe turn, grading is
        # unchanged. Unchanged world + the +500 expectation -> deterministic
        # non-recovery; unchanged world + empty expectations -> recovery.
        monkeypatch.setattr("eval.simulation.runner.create_model", lambda spec: ScriptedAgent())

        sim = RecordingUserSim(generated=0, done_after=99)
        res_false = _runner_with(sim).run(_scenario(self._probe(turn=4)), SPEC)
        assert res_false.probe_fired is True
        assert res_false.recovered is False  # +500 never happened

        sim = RecordingUserSim(generated=0, done_after=99)
        scen_pass = _scenario(self._probe(turn=4), expected_state_changes=[])
        res_true = _runner_with(sim).run(scen_pass, SPEC)
        assert res_true.probe_fired is True
        assert res_true.recovered is True  # unchanged world passes the empty contract

    def test_unfired_rows_excluded_from_recovery_rate(self):
        # The aggregation half of the fix: a declared-but-never-fired row
        # (recovered=None, probe_fired=False) must not enter recovery_rate's
        # numerator OR denominator.
        df = _probe_df({"A": [True, False]})  # 2 scenarios x 2 runs -> rate 0.5
        df["probe_fired"] = True
        unfired = df.iloc[[0]].copy()
        unfired["recovered"] = None
        unfired["probe_fired"] = False
        table = compute_recovery_rates(pd.concat([df, unfired], ignore_index=True))
        assert table["A"]["recovery_rate"] == 0.5  # unchanged by the unfired row
        assert table["A"]["n_probe_rows"] == 4  # denominator counts fired rows only


# --------------------------------------------------------------------------- #
# 4. Hash handling — the #54 lesson
# --------------------------------------------------------------------------- #
class TestHashHandling:
    def test_probeless_scenario_digest_unchanged(self):
        # The canonical dict for a probe-less scenario must NOT contain a
        # recovery_probe key — so its digest is byte-identical to before the
        # field existed (no spurious corpus-hash churn).
        data = _scenario_to_canonical_dict(_scenario(None))
        assert "recovery_probe" not in data

    def test_probe_appears_in_canonical_dict(self):
        probe = RecoveryProbe(turn=4, kind=WRONG_ENTITY, injection="send to BUS-CHK-999")
        data = _scenario_to_canonical_dict(_scenario(probe))
        assert data["recovery_probe"]["kind"] == WRONG_ENTITY
        assert data["recovery_probe"]["turn"] == 4
        assert data["recovery_probe"]["recovery_assertions"] == []

    def test_probe_changes_corpus_hash(self):
        without = {Domain.BANKING: [_scenario(None)]}
        probe = RecoveryProbe(turn=4, kind=WRONG_ENTITY, injection="send to BUS-CHK-999")
        with_probe = {Domain.BANKING: [_scenario(probe)]}
        assert scenario_set_hash(without)[0] != scenario_set_hash(with_probe)[0]

    def test_different_probe_text_changes_hash(self):
        p1 = RecoveryProbe(turn=4, kind=WRONG_ENTITY, injection="send to BUS-CHK-999")
        p2 = RecoveryProbe(turn=4, kind=WRONG_ENTITY, injection="send to BUS-CHK-888")
        h1 = scenario_set_hash({Domain.BANKING: [_scenario(p1)]})[0]
        h2 = scenario_set_hash({Domain.BANKING: [_scenario(p2)]})[0]
        assert h1 != h2

    def test_probeless_digest_matches_legacy_object(self):
        # A Scenario built WITHOUT the recovery_probe attr at all (a legacy
        # object) must hash identically to one with recovery_probe=None.
        from types import SimpleNamespace

        legacy = SimpleNamespace(
            id="banking_probe_0001",
            domain=Domain.BANKING,
            persona={"name": "Margaret"},
            user_goals=["transfer funds", "check balance", "report fraud"],
            tools=[{"name": "initiate_transfer", "description": "move money", "parameters": []}],
            category="adaptive_tool_use",
            initial_message="I need to move some money.",
            ground_truth=_ground_truth(),
            expected_state_changes=_expected_changes(),
        )
        legacy_dict = _scenario_to_canonical_dict(legacy)
        modern_dict = _scenario_to_canonical_dict(_scenario(None))
        assert legacy_dict == modern_dict


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
                {"name": "initiate_transfer", "description": "d", "parameters": []},
                {"name": "get_account_balance", "description": "d", "parameters": []},
            ],
            "initial_message": "hello there I need help",
            "ground_truth": {"accounts": {"BUS-CHK-001": {"balance": 100.0}}},
            "expected_state_changes": [],
        }

    def test_valid_probe_passes(self):
        data = self._base()
        data["recovery_probe"] = {
            "turn": 4,
            "kind": WRONG_ENTITY,
            "injection": "send to BUS-CHK-999",
            "recovery_assertions": [{"assert": "accounts.BUS-CHK-999", "op": "not_exists"}],
        }
        assert validate_scenario_dict(data) == []

    def test_no_probe_unaffected(self):
        assert validate_scenario_dict(self._base()) == []

    def test_bad_kind_rejected(self):
        data = self._base()
        data["recovery_probe"] = {"turn": 4, "kind": "nope", "injection": "x"}
        errs = validate_scenario_dict(data)
        assert any("recovery_probe.kind" in e for e in errs)

    def test_bad_turn_rejected(self):
        data = self._base()
        data["recovery_probe"] = {"turn": 9, "kind": WRONG_ENTITY, "injection": "x"}
        errs = validate_scenario_dict(data)
        assert any("out of range" in e for e in errs)

    def test_empty_injection_rejected(self):
        data = self._base()
        data["recovery_probe"] = {"turn": 4, "kind": WRONG_ENTITY, "injection": ""}
        errs = validate_scenario_dict(data)
        assert any("injection" in e for e in errs)

    def test_bad_recovery_assertion_op_rejected(self):
        data = self._base()
        data["recovery_probe"] = {
            "turn": 4,
            "kind": WRONG_ENTITY,
            "injection": "x",
            "recovery_assertions": [{"assert": "accounts.BUS-CHK-001.balance", "op": "bogus"}],
        }
        errs = validate_scenario_dict(data)
        assert any("unknown op" in e for e in errs)

    def test_not_exists_absent_path_allowed(self):
        # The signature recovery check: assert a wrong entity does NOT exist.
        # not_exists on an absent path is valid (it's how "must not exist" is
        # encoded — the one op that PASSES on an absent path).
        data = self._base()
        data["recovery_probe"] = {
            "turn": 4,
            "kind": WRONG_ENTITY,
            "injection": "x",
            "recovery_assertions": [{"assert": "accounts.BUS-CHK-999", "op": "not_exists"}],
        }
        assert validate_scenario_dict(data) == []


# --------------------------------------------------------------------------- #
# 6. Aggregation: per-model recovery_rate, conditional emission
# --------------------------------------------------------------------------- #
def _probe_df(verdicts_by_model, kinds=None, n_runs=2):
    """Frame from {model: [recovered_bool, ...]} plus optional kinds list."""
    rows = []
    for model, verdicts in verdicts_by_model.items():
        for i, rec in enumerate(verdicts):
            kind = (kinds or [WRONG_ENTITY])[i % len(kinds or [WRONG_ENTITY])]
            for r in range(n_runs):
                rows.append(
                    {
                        "scenario_id": f"probe_scen_{i:02d}",
                        "domain": "banking",
                        "category": "adaptive_tool_use",
                        "model": model,
                        "holdout": False,
                        "sim_profile": "cooperative",
                        "efficacy": 0.9 if rec else 0.2,
                        "task_completion": 0.9,
                        "tool_selection": 0.9,
                        "cost_usd": 0.01,
                        "latency_ms": 2000.0,
                        "total_turns": 7,
                        "reliability_pass_rate": 0.9,
                        "reliability_consistency": 0.9,
                        "tc_agreement": 0.9,
                        "ts_agreement": 0.9,
                        "recovery_probe_kind": kind,
                        "recovered": rec,
                    }
                )
    return pd.DataFrame(rows)


def _plain_df(n_scen=3, n_runs=2):
    """A normal probe-less frame (recovered all-null)."""
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
                    "recovery_probe_kind": None,
                    "recovered": None,
                }
            )
    return pd.DataFrame(rows)


class TestRecoveryAggregation:
    def test_recovery_rate_over_probe_rows(self):
        df = _probe_df({"A": [True, True, False, True]})  # 3/4 recovered
        table = compute_recovery_rates(df)
        assert table["A"]["recovery_rate"] == 0.75
        assert table["A"]["n_probe_scenarios"] == 4
        assert table["A"]["n_probe_rows"] == 8  # 4 scenarios x 2 runs

    def test_by_kind_breakdown(self):
        df = _probe_df(
            {"A": [True, False]},
            kinds=[WRONG_ENTITY, CONTRADICTORY_REFERENCE],
        )
        table = compute_recovery_rates(df)
        assert table["A"]["by_kind"][WRONG_ENTITY]["recovery_rate"] == 1.0
        assert table["A"]["by_kind"][CONTRADICTORY_REFERENCE]["recovery_rate"] == 0.0

    def test_empty_when_no_probe_rows(self):
        assert compute_recovery_rates(_plain_df()) == {}
        assert compute_recovery_rates(pd.DataFrame()) == {}

    def test_missing_column_yields_empty(self):
        df = _plain_df().drop(columns=["recovered"])
        assert compute_recovery_rates(df) == {}

    def test_deterministic(self):
        df = _probe_df({"A": [True, False, True]})
        assert compute_recovery_rates(df) == compute_recovery_rates(df)


class TestRecoveryEmission:
    def test_emitted_when_probe_rows_present(self):
        df = _probe_df({"A": [True, False, True, True]})
        lb = compute_leaderboard(df)
        assert "recovery_probe_robustness" in lb
        assert lb["recovery_probe_robustness"]["models"]["A"]["recovery_rate"] == 0.75

    def test_absent_on_normal_run(self):
        lb = compute_leaderboard(_plain_df())
        assert "recovery_probe_robustness" not in lb

    def test_probe_rows_do_not_move_public_efficacy(self):
        # Mix probe rows (low efficacy) into a frame; the public efficacy is
        # computed over the SAME rows here (probe rows aren't a separate corpus
        # in this synthetic frame), so this asserts the recovery surface is
        # ADDITIVE — present alongside, not instead of, the normal board.
        df = _probe_df({"A": [True, True, True, True]})
        lb = compute_leaderboard(df)
        assert "recovery_probe_robustness" in lb
        assert any(m["name"] == "A" for m in lb["models"])


# --------------------------------------------------------------------------- #
# 7. Demonstration fixtures validate clean and carry the right kinds
# --------------------------------------------------------------------------- #
class TestDemoFixtures:
    def test_fixtures_exist_outside_data_scenarios(self):
        files = sorted(FIXTURE_DIR.glob("*.json"))
        assert len(files) == 3
        # Sanity: they are NOT under data/scenarios.
        for f in files:
            assert "data" not in f.parts or "scenarios" not in f.parts

    @pytest.mark.parametrize(
        "filename,expected_kind",
        [
            ("banking_wrong_entity_probe.json", WRONG_ENTITY),
            ("cs_contradictory_reference_probe.json", CONTRADICTORY_REFERENCE),
            ("banking_incomplete_action_claim_probe.json", INCOMPLETE_ACTION_CLAIM),
        ],
    )
    def test_fixture_validates_and_has_kind(self, filename, expected_kind):
        data = json.loads((FIXTURE_DIR / filename).read_text(encoding="utf-8"))
        errors = validate_scenario_dict(data)
        assert errors == [], f"{filename}: {errors}"
        assert data["recovery_probe"]["kind"] == expected_kind

    def test_one_fixture_per_kind(self):
        kinds = set()
        for f in FIXTURE_DIR.glob("*.json"):
            data = json.loads(f.read_text(encoding="utf-8"))
            kinds.add(data["recovery_probe"]["kind"])
        assert kinds == PROBE_KINDS


# --------------------------------------------------------------------------- #
# 8. Recovery-probe x parameterized-template integration (issues #57 + #60)
# --------------------------------------------------------------------------- #
class TestProbeTemplatingIntegration:
    """A probe on a TEMPLATED scenario must inject the INSTANTIATED entity.

    The load order makes this automatic: scripts/run_eval.py first calls
    instantiate() on the raw on-disk dict (run_eval.py:185/219), and
    eval.templating.substitute() recurses through EVERY field -- including the
    recovery_probe block's injection string and recovery_assertions list -- BEFORE
    _scenario_from_dict() (run_eval.py:160) parses recovery_probe into a
    RecoveryProbe. So by the time RecoveryProbe.from_dict runs, the {{slot}}
    placeholders are already concrete values, and injected_message() returns the
    instantiated text. injected_message() being a method is the documented seam
    for this; here it needs no separate hook because the generic substitution gets
    there first.

    The risk these tests pin: a probe whose injection names an entity (e.g. an
    account id) must NOT reference a stale, never-instantiated id. If the probe
    were parsed before instantiation, the injected message would carry the literal
    {{bad_acct}} (or a published value that does not exist in the per-run world) --
    a silent fault that never lands.
    """

    @staticmethod
    def _templated_probe_scenario() -> dict:
        """A minimal templated banking scenario carrying a wrong_entity probe.

        The probe's injection AND a recovery_assertion path both reference the
        same {{bad_acct}} slot, so both surfaces have to be rewritten coherently
        for the test to pass.
        """
        return {
            "id": "banking_probe_tmpl_0001",
            "category": "adaptive_tool_use",
            "schema_version": "0.2",
            "authorship": {"author_model": "human-handwritten"},
            "template_slots": {
                "good_acct": {"type": "account_id", "prefix": "PERS-CHK-", "length": 4},
                "bad_acct": {"type": "account_id", "prefix": "BUS-CHK-", "length": 4},
            },
            "persona": {
                "name": "Margaret",
                "age": 50,
                "occupation": "teacher",
                "personality_traits": ["careful"],
                "tone": "calm",
                "detail_level": "moderate",
                "background": "manages her own accounts",
            },
            "user_goals": [
                "Verify my identity",
                "Move money into my checking account {{good_acct}}",
                "Confirm the transfer landed",
            ],
            "tools": [
                {"name": "initiate_transfer", "description": "move money", "parameters": []},
                {"name": "get_account_balance", "description": "read balance", "parameters": []},
            ],
            "initial_message": "Please move money into my account {{good_acct}}.",
            "ground_truth": {"accounts": {"{{good_acct}}": {"type": "checking", "balance": 100.0}}},
            "expected_state_changes": [
                {"assert": "accounts.{{good_acct}}.balance", "op": "increased_by", "value": 500.0},
            ],
            "recovery_probe": {
                "turn": 4,
                "kind": WRONG_ENTITY,
                "injection": "Actually, send it to account {{bad_acct}} instead.",
                "recovery_assertions": [
                    {"assert": "accounts.{{bad_acct}}", "op": "not_exists"},
                ],
            },
        }

    def test_injection_carries_instantiated_entity(self):
        raw = self._templated_probe_scenario()
        seed = 4242

        # The same seed resolves bad_acct to a concrete value once; that value is
        # what must appear in the injected message (not the {{bad_acct}} literal).
        mapping = instantiate(raw, seed)  # instantiated dict (template_slots stripped)
        injected_dict = mapping["recovery_probe"]["injection"]

        scenario = _scenario_from_dict(instantiate(raw, seed), Domain.BANKING, holdout=False)
        msg = scenario.recovery_probe.injected_message()

        # The placeholder is gone, and the message matches the instantiated dict.
        assert "{{bad_acct}}" not in msg
        assert "{{" not in msg
        assert msg == injected_dict
        # And it carries a concrete BUS-CHK account id, not a bare prefix / stale id.
        assert "send it to account BUS-CHK-" in msg
        assert msg != "Actually, send it to account BUS-CHK- instead."

    def test_recovery_assertion_path_is_instantiated(self):
        raw = self._templated_probe_scenario()
        seed = 4242

        instantiated = instantiate(raw, seed)
        bad_acct = instantiated["recovery_probe"]["recovery_assertions"][0]["assert"].split(".", 1)[
            1
        ]

        scenario = _scenario_from_dict(instantiated, Domain.BANKING, holdout=False)
        assert_path = scenario.recovery_probe.recovery_assertions[0]["assert"]

        assert "{{bad_acct}}" not in assert_path
        assert assert_path == f"accounts.{bad_acct}"
        # The probe's bad entity matches the id named in the injection -- one slot,
        # substituted coherently across both probe surfaces.
        assert bad_acct in scenario.recovery_probe.injected_message()

    def test_injected_entity_absent_from_world(self):
        # The whole point of a wrong_entity probe: the injected account must not
        # exist in the instantiated ground_truth. A stale (un-instantiated) id
        # could collide or, worse, silently reference nothing meaningful.
        raw = self._templated_probe_scenario()
        instantiated = instantiate(raw, 4242)
        scenario = _scenario_from_dict(instantiated, Domain.BANKING, holdout=False)

        bad_acct = scenario.recovery_probe.recovery_assertions[0]["assert"].split(".", 1)[1]
        good_accts = set(scenario.ground_truth["accounts"])
        assert bad_acct not in good_accts  # the bad entity is genuinely absent
        assert bad_acct in scenario.recovery_probe.injected_message()

    def test_different_seed_changes_injected_entity(self):
        # Determinism + variation: a different run seed yields a different injected
        # account, but the message shape (and probe kind) is invariant.
        raw = self._templated_probe_scenario()
        a = _scenario_from_dict(instantiate(raw, 1), Domain.BANKING, holdout=False)
        b = _scenario_from_dict(instantiate(raw, 2), Domain.BANKING, holdout=False)
        assert a.recovery_probe.injected_message() != b.recovery_probe.injected_message()
        assert a.recovery_probe.kind == b.recovery_probe.kind == WRONG_ENTITY

    def test_instantiated_probe_scenario_validates_clean(self):
        # Invariant 3 of templating (validator-clean output) holds for the probe
        # block too: the instantiated scenario is an ordinary v0.2 scenario whose
        # recovery_probe passes the on-disk validator.
        instantiated = instantiate(self._templated_probe_scenario(), 4242)
        assert validate_scenario_dict(instantiated) == []
