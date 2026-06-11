"""Tests for behavioral user-sim profiles (issue #59, part 1).

Pins the four guarantees of the feature, all fully OFFLINE (stubbed sims, no
API calls, deterministic):

1. **Zero behavior change by default** — with no profile configured (or with
   the explicit ``cooperative`` profile) the user-sim prompt is byte-identical
   to the pre-profile prompt, asserted against a literal snapshot of the
   pre-change constant.
2. **Profiles layer, never replace** — a non-cooperative prompt is exactly the
   default prompt plus an appended behavior block (persona/goals/facts intact).
3. **Provenance everywhere** — ``sim_profile`` is stamped on the
   SimulationResult, the result row, the per-run artifact, the resume
   reconstruction, ``pre_registration.json``, and the run manifest.
4. **No leakage** — non-cooperative rows can never reach the public
   leaderboard aggregates (the tripwire), and the persona-stratified pass-rate
   helper reports them separately and deterministically.
"""

import json

import numpy as np
import pandas as pd
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from eval.artifacts import build_artifact
from eval.config import DEFAULT_SIMULATION, Domain, SimulationConfig
from eval.pre_registration import build_pre_registration
from eval.providers.registry import ModelSpec
from eval.resume import _sim_namespace
from eval.simulation.profiles import (
    ADVERSARIAL_PROFILE,
    CONFUSED_PROFILE,
    COOPERATIVE_PROFILE,
    DEFAULT_SIM_PROFILE,
    IMPATIENT_PROFILE,
    SIM_PROFILES,
    profile_instructions,
)
from eval.simulation.runner import (
    CONVERSATION_COMPLETE,
    ConversationTurn,
    Scenario,
    SimulationRunner,
)
from scripts.aggregate_results import (
    compute_leaderboard,
    compute_sim_profile_pass_rates,
    exclude_non_cooperative_profiles,
)

# --------------------------------------------------------------------------- #
# Snapshot of the user-sim prompt EXACTLY as built before issue #59, for the
# fixed scenario/history below. If the default path ever changes the prompt —
# even by one byte — this test fails. Do not regenerate this constant from the
# code under test; it is the pre-change behavior, written out literally.
# --------------------------------------------------------------------------- #
PRE_PROFILE_DEFAULT_PROMPT = (
    "You are simulating a user in a conversation with an AI agent.\n\n"
    'Persona: {"name": "Test"}\n'
    "Goals (pursue these naturally across the conversation):\n"
    "  1. Check balance\n"
    "\nConversation so far:\n"
    "[user] What is my balance?\n\n"
    "Agent's last response:\n"
    "Hi, how can I help?\n\n"
    "Instructions:\n"
    "- Respond naturally as this persona would\n"
    "- Pursue your remaining unmet goals\n"
    "- If ALL goals have been addressed satisfactorily, respond with "
    'exactly "[CONVERSATION_COMPLETE]"\n'
    "- Do NOT be overly agreeable — push back if the agent's response is "
    "incomplete or unsatisfactory\n"
    "- Keep responses concise (1-3 sentences typically)\n"
)


class RecordingSim(BaseChatModel):
    """A fake simulator that records every prompt it is invoked with."""

    prompts: list = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.prompts.append(messages[0].content)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])

    @property
    def _llm_type(self) -> str:
        return "recording-sim"


def _scenario():
    return Scenario(
        id="test_0001",
        domain=Domain.BANKING,
        persona={"name": "Test"},
        user_goals=["Check balance"],
        tools=[],
        category="adaptive_tool_use",
        initial_message="What is my balance?",
    )


def _captured_prompt(config: SimulationConfig) -> str:
    """Drive _simulate_user_turn with stubbed sims and return the built prompt."""
    runner = SimulationRunner.__new__(SimulationRunner)
    runner.config = config
    sim = RecordingSim(prompts=[])
    runner._user_sim = sim
    runner._tool_sim = sim
    history = [ConversationTurn(turn_number=0, role="user", content="What is my balance?")]
    runner._simulate_user_turn(_scenario(), history, "Hi, how can I help?")
    assert len(sim.prompts) == 1
    return sim.prompts[0]


# --------------------------------------------------------------------------- #
# 1. Profile registry
# --------------------------------------------------------------------------- #
class TestProfileRegistry:
    def test_exactly_four_profiles(self):
        assert set(SIM_PROFILES) == {
            COOPERATIVE_PROFILE,
            IMPATIENT_PROFILE,
            CONFUSED_PROFILE,
            ADVERSARIAL_PROFILE,
        }
        assert DEFAULT_SIM_PROFILE == COOPERATIVE_PROFILE

    def test_cooperative_maps_to_none(self):
        # None == "append nothing" == the byte-identical default prompt.
        assert profile_instructions(COOPERATIVE_PROFILE) is None

    def test_unknown_profile_raises(self):
        with pytest.raises(ValueError, match="Unknown sim profile"):
            profile_instructions("polite-but-firm")

    def test_noncooperative_blocks_carry_issue_behaviors(self):
        # Each profile embeds the behaviors the issue specifies, as exemplars.
        impatient = profile_instructions(IMPATIENT_PROFILE)
        assert "shortcut" in impatient.lower()
        assert "multi-part" in impatient.lower()
        confused = profile_instructions(CONFUSED_PROFILE)
        assert "terminology" in confused.lower()
        assert "verify" in confused.lower()  # wrong-but-confident -> agent verifies
        adversarial = profile_instructions(ADVERSARIAL_PROFILE)
        assert "verification" in adversarial.lower()
        assert "policy" in adversarial.lower()

    def test_blocks_never_override_identity_facts(self):
        # The known-facts invariant: every behavioral block must explicitly keep
        # the identity facts binding (otherwise identity-gated scenarios fail
        # for harness reasons, not agent reasons).
        for name in (IMPATIENT_PROFILE, CONFUSED_PROFILE, ADVERSARIAL_PROFILE):
            block = profile_instructions(name)
            assert "exactly as stated" in block, name

    def test_blocks_format_cleanly_with_complete_token(self):
        for name in (IMPATIENT_PROFILE, CONFUSED_PROFILE, ADVERSARIAL_PROFILE):
            formatted = profile_instructions(name).format(complete_token=CONVERSATION_COMPLETE)
            assert "{complete_token}" not in formatted
        # Impatient references the completion token (it may abandon the chat).
        assert CONVERSATION_COMPLETE in profile_instructions(IMPATIENT_PROFILE).format(
            complete_token=CONVERSATION_COMPLETE
        )


# --------------------------------------------------------------------------- #
# 2. The cooperative default prompt is byte-identical to pre-#59 behavior
# --------------------------------------------------------------------------- #
class TestDefaultPromptUnchanged:
    def test_default_config_prompt_matches_pre_change_snapshot(self):
        # No profile configured anywhere: the prompt must equal the literal
        # pre-change constant, byte for byte.
        assert _captured_prompt(DEFAULT_SIMULATION) == PRE_PROFILE_DEFAULT_PROMPT

    def test_explicit_cooperative_is_identical_to_default(self):
        prompt = _captured_prompt(SimulationConfig(user_sim_profile=COOPERATIVE_PROFILE))
        assert prompt == PRE_PROFILE_DEFAULT_PROMPT


# --------------------------------------------------------------------------- #
# 3. Non-cooperative prompts = default prompt + appended block (layered)
# --------------------------------------------------------------------------- #
class TestProfilePromptLayering:
    @pytest.mark.parametrize("profile", [IMPATIENT_PROFILE, CONFUSED_PROFILE, ADVERSARIAL_PROFILE])
    def test_profile_prompt_is_default_plus_block(self, profile):
        prompt = _captured_prompt(SimulationConfig(user_sim_profile=profile))
        # The full default prompt survives verbatim as a prefix — persona,
        # goals, and instructions are layered on, never replaced.
        assert prompt.startswith(PRE_PROFILE_DEFAULT_PROMPT)
        expected_block = profile_instructions(profile).format(complete_token=CONVERSATION_COMPLETE)
        assert prompt == PRE_PROFILE_DEFAULT_PROMPT + "\n" + expected_block + "\n"
        assert "{complete_token}" not in prompt


# --------------------------------------------------------------------------- #
# 4. Provenance: SimulationResult / row / artifact / resume / pre-registration
# --------------------------------------------------------------------------- #
class ScriptedAgent(BaseChatModel):
    """Fake agent returning one scripted text reply (no tools, no API)."""

    text: str = "Your balance is $500."

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.text))])

    def bind_tools(self, tools, **kwargs):  # noqa: ARG002
        return self

    @property
    def _llm_type(self) -> str:
        return "scripted-agent"


class ConstantSim(BaseChatModel):
    text: str

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.text))])

    @property
    def _llm_type(self) -> str:
        return "constant-sim"


SPEC = ModelSpec(name="FakeModel", model_id="fake", provider="openai")


def _run_result(monkeypatch, config: SimulationConfig):
    runner = SimulationRunner.__new__(SimulationRunner)
    runner.config = config
    runner._user_sim = ConstantSim(text=CONVERSATION_COMPLETE)
    runner._tool_sim = ConstantSim(text="{}")
    monkeypatch.setattr("eval.simulation.runner.create_model", lambda spec: ScriptedAgent())
    return runner.run(_scenario(), SPEC)


class TestProvenanceStamping:
    def test_simulation_result_default_profile(self, monkeypatch):
        result = _run_result(monkeypatch, DEFAULT_SIMULATION)
        assert result.sim_profile == COOPERATIVE_PROFILE

    def test_simulation_result_noncooperative_profile(self, monkeypatch):
        result = _run_result(monkeypatch, SimulationConfig(user_sim_profile=ADVERSARIAL_PROFILE))
        assert result.sim_profile == ADVERSARIAL_PROFILE

    def test_error_path_still_stamped(self, monkeypatch):
        class ExplodingAgent(ScriptedAgent):
            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                raise RuntimeError("provider down")

        runner = SimulationRunner.__new__(SimulationRunner)
        runner.config = SimulationConfig(user_sim_profile=IMPATIENT_PROFILE)
        runner._user_sim = ConstantSim(text=CONVERSATION_COMPLETE)
        runner._tool_sim = ConstantSim(text="{}")
        monkeypatch.setattr("eval.simulation.runner.create_model", lambda spec: ExplodingAgent())
        result = runner.run(_scenario(), SPEC)
        assert result.error is not None
        assert result.sim_profile == IMPATIENT_PROFILE

    def test_runner_init_rejects_unknown_profile(self):
        with pytest.raises(ValueError, match="Unknown sim profile"):
            SimulationRunner(config=SimulationConfig(user_sim_profile="nope"))

    def test_result_row_carries_profile(self, monkeypatch):
        from eval.scoring.judge import ConsensusResult
        from scripts.run_eval import build_result_row

        sim = _run_result(monkeypatch, SimulationConfig(user_sim_profile=CONFUSED_PROFILE))
        consensus = ConsensusResult(
            scenario_id="test_0001",
            rubric_type="task_completion",
            judge_results=[],
            consensus_score=0.8,
            agreement_rate=None,
            max_disagreement=None,
            n_judges_requested=1,
            n_judges_valid=1,
        )
        row = build_result_row(_scenario(), SPEC, sim, consensus, consensus, 0.8, 0.0)
        assert row["sim_profile"] == CONFUSED_PROFILE

    def test_result_row_defaults_cooperative_for_legacy_sim_result(self):
        from types import SimpleNamespace

        from eval.scoring.judge import ConsensusResult
        from scripts.run_eval import build_result_row

        legacy_sim = SimpleNamespace(  # no sim_profile attribute at all
            total_latency_ms=1.0,
            total_turns=1,
            total_input_tokens=1,
            total_output_tokens=1,
            completed=True,
        )
        consensus = ConsensusResult(
            scenario_id="test_0001",
            rubric_type="task_completion",
            judge_results=[],
            consensus_score=0.8,
            agreement_rate=None,
            max_disagreement=None,
            n_judges_requested=1,
            n_judges_valid=1,
        )
        row = build_result_row(_scenario(), SPEC, legacy_sim, consensus, consensus, 0.8, 0.0)
        assert row["sim_profile"] == COOPERATIVE_PROFILE

    def test_artifact_sim_meta_carries_profile(self, monkeypatch):
        from eval.scoring.judge import ConsensusResult

        sim = _run_result(monkeypatch, SimulationConfig(user_sim_profile=ADVERSARIAL_PROFILE))
        consensus = ConsensusResult(
            scenario_id="test_0001",
            rubric_type="task_completion",
            judge_results=[],
            consensus_score=0.8,
            agreement_rate=None,
            max_disagreement=None,
            n_judges_requested=1,
            n_judges_valid=1,
        )
        payload = build_artifact("test_0001", "FakeModel", 0, sim, consensus, consensus)
        assert payload["sim_meta"]["sim_profile"] == ADVERSARIAL_PROFILE

    def test_resume_namespace_reads_profile_with_legacy_fallback(self):
        ns = _sim_namespace({"sim_meta": {"sim_profile": IMPATIENT_PROFILE}})
        assert ns.sim_profile == IMPATIENT_PROFILE
        # Artifacts that predate the field were all cooperative by construction.
        legacy = _sim_namespace({"sim_meta": {"completed": True}})
        assert legacy.sim_profile == COOPERATIVE_PROFILE

    def test_pre_registration_records_profile(self):
        from eval.config import JUDGES

        kwargs = dict(
            run_id="results_test",
            models=[{"name": "M", "model_id": "m-1", "provider": "openai"}],
            scenarios_by_domain={Domain.BANKING: [_scenario()]},
            judges=JUDGES,
            judge_keys=["opus"],
            reliability_runs=1,
            bootstrap_seed=42,
            agent_temperature=0.0,
            user_simulator_temperature=0.7,
            tool_simulator_temperature=0.0,
            separate_judge_calls=False,
        )
        reg = build_pre_registration(**kwargs, user_sim_profile=ADVERSARIAL_PROFILE)
        assert reg["seeds_and_temperatures"]["user_sim_profile"] == ADVERSARIAL_PROFILE
        # Omitted -> the cooperative default, matching the no-flag run.
        reg_default = build_pre_registration(**kwargs)
        assert reg_default["seeds_and_temperatures"]["user_sim_profile"] == COOPERATIVE_PROFILE


# --------------------------------------------------------------------------- #
# 5. run_eval CLI wiring (offline, faked evaluation loop)
# --------------------------------------------------------------------------- #
def _run_cli(tmp_path, monkeypatch, extra_argv):
    """Drive run_eval.main offline; returns (results_dir, output_path)."""
    import scripts.run_eval as run_eval

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    output = results_dir / "results_20260611_000001.parquet"

    monkeypatch.setattr(run_eval, "load_scenarios", lambda domain: [_scenario()])
    monkeypatch.setattr(run_eval, "init_tracing", lambda **kw: None)
    monkeypatch.setattr(run_eval, "get_tracer", lambda: None)
    monkeypatch.setattr(run_eval, "SimulationRunner", lambda *a, **k: object())

    captured = {"profile": None}
    orig_simconfig = run_eval.SimulationConfig

    def capturing_simconfig(**kwargs):
        cfg = orig_simconfig(**kwargs)
        captured["profile"] = cfg.user_sim_profile
        return cfg

    monkeypatch.setattr(run_eval, "SimulationConfig", capturing_simconfig)

    def fake_evaluate(runner, scenario, agent_spec, tracer, judge_keys, **kw):
        from types import SimpleNamespace

        from eval.scoring.judge import ConsensusResult

        sim = SimpleNamespace(
            total_latency_ms=1.0,
            total_turns=1,
            total_input_tokens=10,
            total_output_tokens=5,
            completed=True,
            ended_by="user_sim",
            state_progress_at_end=None,
            premature_end=False,
            resolved_model="gpt-5.5",
            user_sim_model="gpt-4.1-mini-2025-04-14",
            tool_sim_model="gpt-4.1-mini-2025-04-14",
            sim_profile=captured["profile"],
        )
        consensus = ConsensusResult(
            scenario_id=scenario.id,
            rubric_type="task_completion",
            judge_results=[],
            consensus_score=0.8,
            agreement_rate=None,
            max_disagreement=None,
            n_judges_requested=1,
            n_judges_valid=1,
        )
        row = run_eval.build_result_row(scenario, agent_spec, sim, consensus, consensus, 0.8, 0.0)
        return row, 0.0

    monkeypatch.setattr(run_eval, "evaluate_scenario", fake_evaluate)

    argv = [
        "run_eval",
        "--domains",
        "banking",
        "--models",
        "GPT-5.5",
        "--judges",
        "opus",
        "--reliability-runs",
        "1",
        "--no-artifacts",
        "--parallel-models",
        "1",
        "--output",
        str(output),
        *extra_argv,
    ]
    monkeypatch.setattr("sys.argv", argv)
    monkeypatch.chdir(tmp_path)
    run_eval.main()
    return results_dir, output


class TestRunEvalWiring:
    def test_sim_profile_flag_lands_everywhere(self, tmp_path, monkeypatch):
        results_dir, output = _run_cli(
            tmp_path, monkeypatch, ["--sim-profile", ADVERSARIAL_PROFILE]
        )
        reg = json.loads((results_dir / "pre_registration.json").read_text(encoding="utf-8"))
        assert reg["seeds_and_temperatures"]["user_sim_profile"] == ADVERSARIAL_PROFILE

        manifest = json.loads((results_dir / "run_manifest.json").read_text(encoding="utf-8"))
        assert manifest["sim_profile"] == ADVERSARIAL_PROFILE

        df = pd.read_parquet(output)
        assert set(df["sim_profile"]) == {ADVERSARIAL_PROFILE}

    def test_default_is_cooperative_everywhere(self, tmp_path, monkeypatch):
        results_dir, output = _run_cli(tmp_path, monkeypatch, [])
        reg = json.loads((results_dir / "pre_registration.json").read_text(encoding="utf-8"))
        assert reg["seeds_and_temperatures"]["user_sim_profile"] == COOPERATIVE_PROFILE
        manifest = json.loads((results_dir / "run_manifest.json").read_text(encoding="utf-8"))
        assert manifest["sim_profile"] == COOPERATIVE_PROFILE
        df = pd.read_parquet(output)
        assert set(df["sim_profile"]) == {COOPERATIVE_PROFILE}


# --------------------------------------------------------------------------- #
# 6. Aggregation: exclusion tripwire + persona-stratified pass rates
# --------------------------------------------------------------------------- #
def _profile_df(spec, n_scen=4, n_runs=2, seed=7):
    """Results frame from {model: {profile: base_efficacy}} (test_holdout style)."""
    rng = np.random.default_rng(seed)
    rows = []
    for model, by_profile in spec.items():
        for profile, base in by_profile.items():
            for s in range(n_scen):
                for r in range(n_runs):
                    eff = float(np.clip(base + rng.normal(0, 0.02), 0, 1))
                    rows.append(
                        {
                            "scenario_id": f"scen_{s:02d}",
                            "domain": "banking",
                            "category": "adaptive_tool_use",
                            "model": model,
                            "holdout": False,
                            "sim_profile": profile,
                            "efficacy": eff,
                            "task_completion": eff,
                            "tool_selection": eff,
                            "cost_usd": 0.01,
                            "latency_ms": 2000.0,
                            "total_turns": 5,
                            "reliability_pass_rate": base,
                            "reliability_consistency": 0.9,
                            "tc_agreement": 0.9,
                            "ts_agreement": 0.9,
                        }
                    )
    return pd.DataFrame(rows)


class TestExclusion:
    def test_drops_only_noncooperative_rows(self):
        df = _profile_df({"A": {COOPERATIVE_PROFILE: 0.9, ADVERSARIAL_PROFILE: 0.2}})
        out = exclude_non_cooperative_profiles(df)
        assert set(out["sim_profile"]) == {COOPERATIVE_PROFILE}
        assert len(out) == len(df) / 2

    def test_missing_column_keeps_everything(self):
        # Legacy parquets (pre-#59) have no sim_profile column: all cooperative.
        df = _profile_df({"A": {COOPERATIVE_PROFILE: 0.9}}).drop(columns=["sim_profile"])
        assert len(exclude_non_cooperative_profiles(df)) == len(df)

    def test_null_profile_counts_as_cooperative(self):
        df = _profile_df({"A": {COOPERATIVE_PROFILE: 0.9}})
        df.loc[df.index[:2], "sim_profile"] = None
        assert len(exclude_non_cooperative_profiles(df)) == len(df)


class TestLeaderboardTripwire:
    def test_noncooperative_rows_cannot_move_public_efficacy(self):
        # Adversarial rows score 0.1; if they leaked, A's efficacy would blend
        # down toward 0.5. The public board must read ~0.9 (cooperative only).
        df = _profile_df(
            {
                "A": {COOPERATIVE_PROFILE: 0.9, ADVERSARIAL_PROFILE: 0.1},
                "B": {COOPERATIVE_PROFILE: 0.5},
            }
        )
        lb = compute_leaderboard(df)
        entry_a = next(m for m in lb["models"] if m["name"] == "A")
        assert abs(entry_a["efficacy"] - 0.9) < 0.05

    def test_no_profile_name_appears_in_leaderboard_json(self):
        df = _profile_df(
            {
                "A": {
                    COOPERATIVE_PROFILE: 0.9,
                    ADVERSARIAL_PROFILE: 0.1,
                    IMPATIENT_PROFILE: 0.3,
                    CONFUSED_PROFILE: 0.4,
                }
            }
        )
        flat = json.dumps(compute_leaderboard(df))
        for profile in (ADVERSARIAL_PROFILE, IMPATIENT_PROFILE, CONFUSED_PROFILE):
            assert profile not in flat

    def test_noncooperative_only_run_yields_empty_board(self):
        df = _profile_df({"A": {ADVERSARIAL_PROFILE: 0.6}})
        lb = compute_leaderboard(df)
        assert lb["models"] == []


class TestSimProfilePassRates:
    def test_pass_rates_and_delta(self):
        # Deterministic frame: cooperative all-pass, adversarial all-fail.
        df = _profile_df({"A": {COOPERATIVE_PROFILE: 0.95, ADVERSARIAL_PROFILE: 0.10}}, seed=11)
        table = compute_sim_profile_pass_rates(df)
        coop = table["A"][COOPERATIVE_PROFILE]
        adv = table["A"][ADVERSARIAL_PROFILE]
        assert coop["pass_rate"] == 1.0
        assert adv["pass_rate"] == 0.0
        # delta = cooperative - profile: positive == worse under the profile,
        # i.e. the cooperative-only inflation made visible.
        assert adv["delta_vs_cooperative"] == 1.0
        assert coop["delta_vs_cooperative"] is None
        assert coop["n_rows"] == 8
        assert coop["n_scenarios"] == 4

    def test_deterministic(self):
        df = _profile_df({"A": {COOPERATIVE_PROFILE: 0.8, IMPATIENT_PROFILE: 0.6}})
        assert compute_sim_profile_pass_rates(df) == compute_sim_profile_pass_rates(df)

    def test_legacy_frame_counts_as_cooperative(self):
        df = _profile_df({"A": {COOPERATIVE_PROFILE: 0.9}}).drop(columns=["sim_profile"])
        table = compute_sim_profile_pass_rates(df)
        assert set(table["A"]) == {COOPERATIVE_PROFILE}

    def test_no_cooperative_rows_yields_null_delta(self):
        df = _profile_df({"A": {ADVERSARIAL_PROFILE: 0.9}})
        table = compute_sim_profile_pass_rates(df)
        assert table["A"][ADVERSARIAL_PROFILE]["delta_vs_cooperative"] is None

    def test_empty_frame(self):
        assert compute_sim_profile_pass_rates(pd.DataFrame()) == {}
