"""Tests for the do-nothing ("null") agent anti-gaming check (issue #28).

Motivation: the Berkeley RDI audit showed trivial agents can game agentic
benchmarks to near-perfect scores. These tests prove that COT Bench's null agent
(a) makes NO API calls and NO tool calls, (b) produces a well-formed transcript
so judges and state checks can still score it, (c) scores 0.0 on the
deterministic state grader for a real state-changing scenario, and (d) can never
appear on the published leaderboard.

All tests are fully OFFLINE: no OpenRouter/Anthropic/OpenAI calls. The user and
tool simulators are stubbed; the null agent is deterministic by construction.
"""

import json
from pathlib import Path

import pandas as pd
from langchain_core.messages import HumanMessage

from eval.config import Domain
from eval.providers.null_agent import (
    NULL_AGENT_NAME,
    NULL_AGENT_PROVIDER,
    NULL_AGENT_RESPONSE,
    NullAgentChatModel,
    create_null_agent,
)
from eval.providers.registry import ModelSpec, create_model
from eval.scoring.state_check import score_state_changes
from eval.simulation.runner import Scenario, SimulationRunner
from scripts.aggregate_results import compute_leaderboard, exclude_non_contestants

# A real state-changing scenario from the corpus. The null agent never calls a
# tool, so the world stays at ground_truth and every expected change fails.
REAL_SCENARIO_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "scenarios"
    / "banking"
    / "banking_adaptive_tool_use_0001.json"
)


# --------------------------------------------------------------------------- #
# Offline simulator stub (reused from the simulation-test pattern)
# --------------------------------------------------------------------------- #
from langchain_core.language_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402

from eval.config import DEFAULT_SIMULATION  # noqa: E402
from eval.simulation.runner import CONVERSATION_COMPLETE  # noqa: E402


class ConstantSim(BaseChatModel):
    """A fake simulator that always returns the same content (no API)."""

    text: str

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.text))])

    @property
    def _llm_type(self) -> str:
        return "constant-sim"


NULL_SPEC = ModelSpec(name=NULL_AGENT_NAME, model_id=NULL_AGENT_NAME, provider=NULL_AGENT_PROVIDER)


# --------------------------------------------------------------------------- #
# (1) The model itself: deterministic, no tool calls, no spend
# --------------------------------------------------------------------------- #
class TestNullAgentModel:
    def test_invoke_returns_trivial_message_no_tool_calls(self):
        model = NullAgentChatModel()
        resp = model.invoke([HumanMessage(content="transfer $5000 now please")])
        assert isinstance(resp, AIMessage)
        assert resp.content == NULL_AGENT_RESPONSE
        assert resp.tool_calls == []

    def test_zero_token_usage(self):
        model = NullAgentChatModel()
        resp = model.invoke([HumanMessage(content="anything")])
        usage = resp.usage_metadata
        assert usage["input_tokens"] == 0
        assert usage["output_tokens"] == 0

    def test_deterministic_across_invocations(self):
        model = NullAgentChatModel()
        a = model.invoke([HumanMessage(content="do X")]).content
        b = model.invoke([HumanMessage(content="do something completely different")]).content
        assert a == b == NULL_AGENT_RESPONSE

    def test_bind_tools_is_noop_returns_self(self):
        model = NullAgentChatModel()
        bound = model.bind_tools([{"type": "function", "function": {"name": "transfer"}}])
        assert bound is model
        # Even with tools "bound", it still emits no tool calls.
        assert bound.invoke([HumanMessage(content="use the transfer tool")]).tool_calls == []

    def test_registry_resolves_null_provider(self):
        model = create_model(NULL_SPEC)
        assert isinstance(model, NullAgentChatModel)

    def test_factory_ignores_spec(self):
        assert isinstance(create_null_agent(NULL_SPEC), NullAgentChatModel)


# --------------------------------------------------------------------------- #
# (2) End-to-end simulation: well-formed transcript, no tools, world unchanged
# --------------------------------------------------------------------------- #
def _make_null_runner(monkeypatch):
    """A runner whose simulators are stubbed and whose agent is the null agent."""
    runner = SimulationRunner.__new__(SimulationRunner)
    runner.config = DEFAULT_SIMULATION
    runner._user_sim = ConstantSim(text=CONVERSATION_COMPLETE)
    runner._tool_sim = ConstantSim(text="{}")
    # The runner builds the agent via create_model; for the null provider that
    # already returns a fresh NullAgentChatModel — but patch to be explicit and
    # guarantee no other provider is ever constructed.
    monkeypatch.setattr(
        "eval.simulation.runner.create_model",
        lambda spec: NullAgentChatModel(),
    )
    return runner


def _load_real_scenario() -> Scenario:
    data = json.loads(REAL_SCENARIO_PATH.read_text())
    return Scenario(
        id=data["id"],
        domain=Domain.BANKING,
        persona=data["persona"],
        user_goals=data["user_goals"],
        tools=data["tools"],
        category=data["category"],
        initial_message=data["initial_message"],
        ground_truth=data.get("ground_truth"),
        expected_state_changes=data.get("expected_state_changes"),
    )


class TestNullAgentSimulation:
    def test_transcript_is_well_formed_with_no_tool_calls(self, monkeypatch):
        runner = _make_null_runner(monkeypatch)
        result = runner.run(_load_real_scenario(), NULL_SPEC)

        # There is at least one user turn and one agent turn (well-formed).
        agent_turns = [t for t in result.turns if t.role == "agent"]
        assert agent_turns, "null agent produced no agent turn"
        # Every agent turn is the trivial reply with NO tool calls.
        for t in agent_turns:
            assert t.content == NULL_AGENT_RESPONSE
            assert t.tool_calls == []
        # No tool turns at all — the agent never invoked a tool.
        assert not any(t.role == "tool" for t in result.turns)

    def test_no_spend_and_no_error(self, monkeypatch):
        runner = _make_null_runner(monkeypatch)
        result = runner.run(_load_real_scenario(), NULL_SPEC)
        assert result.error is None
        assert result.total_input_tokens == 0
        assert result.total_output_tokens == 0

    def test_world_unchanged_after_do_nothing(self, monkeypatch):
        runner = _make_null_runner(monkeypatch)
        scenario = _load_real_scenario()
        result = runner.run(scenario, NULL_SPEC)
        # No tool calls means no state_delta ever applied; final == initial.
        assert result.final_world == scenario.ground_truth


# --------------------------------------------------------------------------- #
# (3) Deterministic state grader scores the do-nothing run 0.0 on a REAL scenario
# --------------------------------------------------------------------------- #
class TestNullAgentStateScoreIsZero:
    def test_state_grader_gives_zero_on_real_scenario(self, monkeypatch):
        runner = _make_null_runner(monkeypatch)
        scenario = _load_real_scenario()
        result = runner.run(scenario, NULL_SPEC)

        state = score_state_changes(
            scenario.ground_truth,
            result.final_world,
            scenario.expected_state_changes,
        )
        assert state is not None
        # Every expected state change failed — a do-nothing agent scores 0.0.
        assert state["score"] == 0.0
        assert state["n_passed"] == 0
        assert state["n_total"] == len(scenario.expected_state_changes)

    def test_scenario_actually_expects_state_changes(self):
        # Guard: this proof is only meaningful if the chosen scenario expects
        # non-trivial state changes (a non-empty assertion list).
        scenario = _load_real_scenario()
        assert scenario.expected_state_changes
        assert len(scenario.expected_state_changes) >= 3


# --------------------------------------------------------------------------- #
# (4) The null agent can NEVER appear on the published leaderboard
# --------------------------------------------------------------------------- #
def _result_row(model: str, **over) -> dict:
    """A minimal results row with the columns compute_leaderboard reads."""
    row = {
        "scenario_id": "banking_adaptive_tool_use_0001",
        "domain": "banking",
        "model": model,
        "efficacy": 0.0,
        "task_completion": 0.0,
        "tool_selection": 0.0,
        "state_score": 0.0,
        "cost_usd": 0.0,
        "latency_ms": 1.0,
        "total_turns": 1,
        "reliability_pass_rate": 0.0,
        "reliability_consistency": 0.0,
        "tc_agreement": None,
        "ts_agreement": None,
    }
    row.update(over)
    return row


class TestLeaderboardExcludesNullAgent:
    def test_exclude_non_contestants_drops_null_agent(self):
        df = pd.DataFrame(
            [
                _result_row("GPT-5.5", efficacy=0.8),
                _result_row(NULL_AGENT_NAME),
                _result_row("Claude Sonnet 4.6", efficacy=0.7),
            ]
        )
        kept = exclude_non_contestants(df)
        assert NULL_AGENT_NAME not in set(kept["model"])
        assert set(kept["model"]) == {"GPT-5.5", "Claude Sonnet 4.6"}

    def test_exclude_is_case_insensitive(self):
        df = pd.DataFrame([_result_row("GPT-5.5"), _result_row("Null-Agent")])
        kept = exclude_non_contestants(df)
        assert set(kept["model"]) == {"GPT-5.5"}

    def test_compute_leaderboard_omits_null_agent(self):
        df = pd.DataFrame(
            [
                _result_row("GPT-5.5", efficacy=0.8),
                _result_row("Claude Sonnet 4.6", efficacy=0.7),
                _result_row(NULL_AGENT_NAME, efficacy=0.0),
            ]
        )
        board = compute_leaderboard(df)
        names = {m["name"] for m in board["models"]}
        assert NULL_AGENT_NAME not in names
        assert names == {"GPT-5.5", "Claude Sonnet 4.6"}

    def test_null_agent_not_in_models_under_test(self):
        from eval.config import MODELS_UNDER_TEST

        assert all(m["name"] != NULL_AGENT_NAME for m in MODELS_UNDER_TEST)
