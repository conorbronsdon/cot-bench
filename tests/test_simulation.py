"""Tests for the simulation runner — native tool calling, agent/tool loop,
transcript ordering, schema conversion, and token accounting.

All chat models are stubbed; NO real API calls are made.
"""

from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from eval.config import DEFAULT_SIMULATION, Domain
from eval.providers.registry import ModelSpec
from eval.simulation.runner import (
    _TOOL_CALL_RE,
    CONVERSATION_COMPLETE,
    MAX_TOOL_ROUNDS_PER_TURN,
    ConversationTurn,
    Scenario,
    SimulationRunner,
    ToolCall,
    apply_state_delta,
    tool_to_json_schema,
)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class ScriptedAgent(BaseChatModel):
    """A fake agent chat model that returns a scripted sequence of AIMessages.

    Each invoke() pops the next scripted message. bind_tools() is a no-op that
    returns self so the runner's native-tool-calling path works without an API.
    """

    responses: list[AIMessage]
    cursor: int = 0
    invoke_count: int = 0

    def _generate(
        self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs: Any
    ) -> ChatResult:
        idx = min(self.cursor, len(self.responses) - 1)
        msg = self.responses[idx]
        self.cursor += 1
        self.invoke_count += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools, **kwargs):  # noqa: ARG002
        return self

    @property
    def _llm_type(self) -> str:
        return "scripted-agent"


class ConstantSim(BaseChatModel):
    """A fake simulator that always returns the same content."""

    text: str

    def _generate(
        self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs: Any
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.text))])

    @property
    def _llm_type(self) -> str:
        return "constant-sim"


def _ai_with_tool_call(name: str, args: dict, call_id: str, input_t=10, output_t=5) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id}],
        usage_metadata={"input_tokens": input_t, "output_tokens": output_t, "total_tokens": 0},
    )


def _ai_text(content: str, input_t=8, output_t=4) -> AIMessage:
    return AIMessage(
        content=content,
        usage_metadata={"input_tokens": input_t, "output_tokens": output_t, "total_tokens": 0},
    )


def _make_runner(agent_responses, tool_text="{}", user_text=CONVERSATION_COMPLETE):
    """Build a SimulationRunner with stubbed sim models and a scripted agent.

    Returns (runner, agent_model). create_model is monkeypatched at module level
    by the caller; here we just wire the simulator stubs directly.
    """
    runner = SimulationRunner.__new__(SimulationRunner)
    runner.config = DEFAULT_SIMULATION
    runner._user_sim = ConstantSim(text=user_text)
    runner._tool_sim = ConstantSim(text=tool_text)
    agent = ScriptedAgent(responses=agent_responses)
    return runner, agent


def _scenario():
    return Scenario(
        id="test_0001",
        domain=Domain.BANKING,
        persona={"name": "Test"},
        user_goals=["Check balance"],
        tools=[
            {
                "name": "get_balance",
                "description": "Get balance",
                "parameters": [
                    {"name": "account_id", "type": "string", "required": True},
                ],
            }
        ],
        category="adaptive_tool_use",
        initial_message="What is my balance?",
    )


def _patch_create_model(monkeypatch, agent):
    """Patch create_model so runner.run() uses our scripted agent."""
    monkeypatch.setattr("eval.simulation.runner.create_model", lambda spec: agent)


SPEC = ModelSpec(name="FakeModel", model_id="fake", provider="openai")


# --------------------------------------------------------------------------- #
# (a) agent re-invoked after tool results; follow-up appears in transcript
# (b) transcript ordering: user -> agent -> tool -> agent
# --------------------------------------------------------------------------- #
class TestAgentToolLoop:
    def test_agent_reinvoked_and_followup_in_transcript(self, monkeypatch):
        agent = ScriptedAgent(
            responses=[
                _ai_with_tool_call("get_balance", {"account_id": "A1"}, "call_1"),
                _ai_text("Your balance is $500."),
            ]
        )
        runner, _ = _make_runner([])
        _patch_create_model(monkeypatch, agent)

        result = runner.run(_scenario(), SPEC)

        # Agent invoked twice: once for the tool call, once for the follow-up.
        assert agent.invoke_count == 2
        agent_turns = [t for t in result.turns if t.role == "agent"]
        assert len(agent_turns) == 2
        assert agent_turns[1].content == "Your balance is $500."

    def test_transcript_ordering(self, monkeypatch):
        agent = ScriptedAgent(
            responses=[
                _ai_with_tool_call("get_balance", {"account_id": "A1"}, "call_1"),
                _ai_text("Your balance is $500."),
            ]
        )
        runner, _ = _make_runner([])
        _patch_create_model(monkeypatch, agent)

        result = runner.run(_scenario(), SPEC)
        roles = [t.role for t in result.turns]

        # user -> agent (tool call) -> tool -> agent (follow-up)
        assert roles == ["user", "agent", "tool", "agent"]
        # The agent turn that issued the call comes BEFORE the tool result.
        agent_idx = roles.index("agent")
        tool_idx = roles.index("tool")
        assert agent_idx < tool_idx
        assert result.turns[agent_idx].tool_calls[0].tool_name == "get_balance"
        # The tool turn carries the matching tool_call_id.
        assert result.turns[tool_idx].tool_call_id == "call_1"


# --------------------------------------------------------------------------- #
# (c) inner tool-round cap terminates a runaway tool-calling loop
# --------------------------------------------------------------------------- #
class TestInnerCap:
    def test_cap_terminates_loop(self, monkeypatch):
        # Agent ALWAYS calls a tool, never produces a tool-free message.
        always_tool = [
            _ai_with_tool_call("get_balance", {"account_id": "A1"}, f"call_{i}") for i in range(50)
        ]
        agent = ScriptedAgent(responses=always_tool)
        runner, _ = _make_runner([])
        _patch_create_model(monkeypatch, agent)

        result = runner.run(_scenario(), SPEC)

        # One user turn (user sim says COMPLETE after). The inner loop runs at
        # most MAX_TOOL_ROUNDS_PER_TURN + 1 agent invocations for that turn.
        assert agent.invoke_count == MAX_TOOL_ROUNDS_PER_TURN + 1
        # It terminated rather than looping forever.
        assert result.total_turns == MAX_TOOL_ROUNDS_PER_TURN + 1


# --------------------------------------------------------------------------- #
# (d) schema conversion: valid JSON Schema incl. enum/required handling
# --------------------------------------------------------------------------- #
class TestSchemaConversion:
    def test_basic_conversion(self):
        tool = {
            "name": "get_account_balance",
            "description": "Get current balance",
            "parameters": [
                {"name": "account_id", "type": "string", "required": True},
                {
                    "name": "account_type",
                    "type": "string",
                    "required": False,
                    "enum": ["checking", "savings"],
                },
                {"name": "limit", "type": "integer", "required": False},
            ],
        }
        schema = tool_to_json_schema(tool)

        assert schema["type"] == "function"
        fn = schema["function"]
        assert fn["name"] == "get_account_balance"
        assert fn["description"] == "Get current balance"

        params = fn["parameters"]
        assert params["type"] == "object"
        props = params["properties"]
        assert props["account_id"]["type"] == "string"
        assert props["limit"]["type"] == "integer"
        # enum preserved
        assert props["account_type"]["enum"] == ["checking", "savings"]
        # required array only includes required params
        assert params["required"] == ["account_id"]
        assert "account_type" not in params.get("required", [])

    def test_no_required_omits_required_array(self):
        tool = {
            "name": "ping",
            "description": "ping",
            "parameters": [{"name": "x", "type": "string", "required": False}],
        }
        schema = tool_to_json_schema(tool)
        assert "required" not in schema["function"]["parameters"]

    def test_unknown_type_defaults_to_string(self):
        tool = {
            "name": "t",
            "description": "",
            "parameters": [{"name": "p", "type": "weirdtype", "required": True}],
        }
        schema = tool_to_json_schema(tool)
        assert schema["function"]["parameters"]["properties"]["p"]["type"] == "string"


# --------------------------------------------------------------------------- #
# (e) token totals sum across inner rounds
# --------------------------------------------------------------------------- #
class TestTokenAccounting:
    def test_tokens_sum_across_inner_rounds(self, monkeypatch):
        agent = ScriptedAgent(
            responses=[
                _ai_with_tool_call(
                    "get_balance", {"account_id": "A1"}, "call_1", input_t=10, output_t=5
                ),
                _ai_text("Your balance is $500.", input_t=8, output_t=4),
            ]
        )
        runner, _ = _make_runner([])
        _patch_create_model(monkeypatch, agent)

        result = runner.run(_scenario(), SPEC)

        # Both invocations counted: 10+8 input, 5+4 output.
        assert result.total_input_tokens == 18
        assert result.total_output_tokens == 9


# --------------------------------------------------------------------------- #
# Fallback regex (kept as labeled fallback) still parses legacy embedded calls
# --------------------------------------------------------------------------- #
class TestFallbackRegex:
    def test_simple_tool_call(self):
        content = '{"tool_call": {"name": "get_balance", "arguments": {"account_id": "123"}}}'
        assert len(_TOOL_CALL_RE.findall(content)) == 1

    def test_nested_arguments(self):
        content = (
            '{"tool_call": {"name": "create_transfer", '
            '"arguments": {"from": {"id": "A1", "type": "checking"}, "amount": 500}}}'
        )
        assert len(_TOOL_CALL_RE.findall(content)) == 1

    def test_no_tool_call(self):
        assert len(_TOOL_CALL_RE.findall("Just a normal message.")) == 0

    def test_fallback_fires_when_no_native_calls(self, monkeypatch):
        # Agent returns NO native tool_calls, only embedded JSON in content,
        # then a follow-up. Exercises the labeled fallback path.
        embedded = AIMessage(
            content='{"tool_call": {"name": "get_balance", "arguments": {"account_id": "A1"}}}',
            usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )
        followup = _ai_text("Balance is $500.")
        agent = ScriptedAgent(responses=[embedded, followup])
        runner, _ = _make_runner([])
        _patch_create_model(monkeypatch, agent)

        result = runner.run(_scenario(), SPEC)
        agent_turns = [t for t in result.turns if t.role == "agent"]
        # First agent turn recovered the embedded call.
        assert agent_turns[0].tool_calls[0].tool_name == "get_balance"
        assert any(t.role == "tool" for t in result.turns)


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
class TestDataStructures:
    def test_tool_call_fields(self):
        tc = ToolCall(turn=0, tool_name="test", arguments={"a": 1}, result="ok")
        assert tc.turn == 0
        assert tc.tool_name == "test"
        assert tc.tool_call_id == ""

    def test_conversation_turn_defaults(self):
        turn = ConversationTurn(turn_number=0, role="user", content="hi")
        assert turn.tool_calls == []
        assert turn.latency_ms == 0.0
        assert turn.token_count == 0
        assert turn.tool_call_id == ""


# --------------------------------------------------------------------------- #
# apply_state_delta: nested set, __append__, invalid path skip
# --------------------------------------------------------------------------- #
class TestApplyStateDelta:
    def test_nested_set(self):
        world = {"accounts": {"A1": {"balance": 100.0}}}
        apply_state_delta(world, {"accounts.A1.balance": 150.0})
        assert world["accounts"]["A1"]["balance"] == 150.0

    def test_creates_intermediate_dicts(self):
        world = {}
        apply_state_delta(world, {"a.b.c": 7})
        assert world == {"a": {"b": {"c": 7}}}

    def test_append_to_existing_list(self):
        world = {"transfers": [{"id": 1}]}
        apply_state_delta(world, {"transfers": {"__append__": {"id": 2}}})
        assert world["transfers"] == [{"id": 1}, {"id": 2}]

    def test_append_creates_list_when_absent(self):
        world = {}
        apply_state_delta(world, {"fraud_cases": {"__append__": {"txn": "T1"}}})
        assert world["fraud_cases"] == [{"txn": "T1"}]

    def test_plain_value_replaces_list(self):
        world = {"x": [1, 2, 3]}
        apply_state_delta(world, {"x": []})
        assert world["x"] == []

    def test_invalid_path_descend_into_scalar_is_skipped(self):
        world = {"a": 5}
        apply_state_delta(world, {"a.b": 9})
        # 'a' is a scalar; the delta is skipped, world untouched, no crash.
        assert world == {"a": 5}

    def test_append_to_non_list_is_skipped(self):
        world = {"x": {"not": "a list"}}
        apply_state_delta(world, {"x": {"__append__": 1}})
        assert world == {"x": {"not": "a list"}}

    def test_non_dict_delta_ignored(self):
        world = {"a": 1}
        apply_state_delta(world, ["not", "a", "dict"])
        assert world == {"a": 1}


def _stateful_scenario():
    """A scenario carrying ground_truth so the stateful tool path engages."""
    return Scenario(
        id="state_0001",
        domain=Domain.BANKING,
        persona={"name": "Test"},
        user_goals=["Transfer money"],
        tools=[
            {
                "name": "transfer",
                "description": "Transfer funds",
                "parameters": [
                    {"name": "amount", "type": "number", "required": True},
                ],
            }
        ],
        category="adaptive_tool_use",
        initial_message="Move $500 from checking to savings.",
        ground_truth={
            "accounts": {
                "CHK": {"balance": 1000.0},
                "SAV": {"balance": 0.0},
            },
            "transfers": [],
        },
        expected_state_changes=[
            {"assert": "accounts.SAV.balance", "op": "increased_by", "value": 500.0},
        ],
    )


# --------------------------------------------------------------------------- #
# Stateful tool simulation: world mutates via state_delta; final_world captured
# --------------------------------------------------------------------------- #
class TestStatefulSimulation:
    def test_world_mutates_and_final_world_captured(self, monkeypatch):
        agent = ScriptedAgent(
            responses=[
                _ai_with_tool_call("transfer", {"amount": 500}, "call_1"),
                _ai_text("Done, $500 moved."),
            ]
        )
        # Tool-sim returns a structured response + state_delta that mutates SAV.
        delta_json = (
            '{"response": {"status": "ok"}, '
            '"state_delta": {"accounts.SAV.balance": 500.0, '
            '"accounts.CHK.balance": 500.0, '
            '"transfers": {"__append__": {"amount": 500}}}}'
        )
        runner, _ = _make_runner([], tool_text=delta_json)
        _patch_create_model(monkeypatch, agent)

        result = runner.run(_stateful_scenario(), SPEC)

        assert result.final_world is not None
        assert result.final_world["accounts"]["SAV"]["balance"] == 500.0
        assert result.final_world["accounts"]["CHK"]["balance"] == 500.0
        assert result.final_world["transfers"] == [{"amount": 500}]

    def test_agent_sees_only_response_not_delta(self, monkeypatch):
        agent = ScriptedAgent(
            responses=[
                _ai_with_tool_call("transfer", {"amount": 500}, "call_1"),
                _ai_text("Done."),
            ]
        )
        delta_json = (
            '{"response": {"status": "ok", "confirmation": "CONF-1"}, '
            '"state_delta": {"accounts.SAV.balance": 500.0}}'
        )
        runner, _ = _make_runner([], tool_text=delta_json)
        _patch_create_model(monkeypatch, agent)

        result = runner.run(_stateful_scenario(), SPEC)
        tool_turn = next(t for t in result.turns if t.role == "tool")
        # The tool result is the serialized response only — no state_delta leak.
        assert "state_delta" not in tool_turn.content
        assert "CONF-1" in tool_turn.content

    def test_unparseable_tool_sim_does_not_crash(self, monkeypatch):
        agent = ScriptedAgent(
            responses=[
                _ai_with_tool_call("transfer", {"amount": 500}, "call_1"),
                _ai_text("Done."),
            ]
        )
        runner, _ = _make_runner([], tool_text="this is not json at all")
        _patch_create_model(monkeypatch, agent)

        result = runner.run(_stateful_scenario(), SPEC)
        # World unchanged (no delta applied), raw text fed back, run completes.
        assert result.final_world["accounts"]["SAV"]["balance"] == 0.0
        tool_turn = next(t for t in result.turns if t.role == "tool")
        assert "not json" in tool_turn.content

    def test_parse_failure_counted_on_result(self, monkeypatch):
        # S3: an unparseable tool-sim response increments the per-run counter and
        # surfaces it on the SimulationResult.
        agent = ScriptedAgent(
            responses=[
                _ai_with_tool_call("transfer", {"amount": 500}, "call_1"),
                _ai_text("Done."),
            ]
        )
        runner, _ = _make_runner([], tool_text="this is not json at all")
        _patch_create_model(monkeypatch, agent)

        result = runner.run(_stateful_scenario(), SPEC)
        assert result.tool_sim_parse_failures == 1

    def test_clean_run_zero_parse_failures(self, monkeypatch):
        # S3: a parseable tool-sim response leaves the counter at 0.
        agent = ScriptedAgent(
            responses=[
                _ai_with_tool_call("transfer", {"amount": 500}, "call_1"),
                _ai_text("Done."),
            ]
        )
        delta_json = (
            '{"response": {"status": "ok"}, "state_delta": {"accounts.SAV.balance": 500.0}}'
        )
        runner, _ = _make_runner([], tool_text=delta_json)
        _patch_create_model(monkeypatch, agent)

        result = runner.run(_stateful_scenario(), SPEC)
        assert result.tool_sim_parse_failures == 0


# --------------------------------------------------------------------------- #
# Coded tool transitions wired into the runner (issue #87, phase 1b)
# --------------------------------------------------------------------------- #
def _coded_transfer_scenario():
    """Stateful banking scenario whose tool (``initiate_transfer``) is a REGISTERED
    coded transition, so the runner mutates the world deterministically without
    ever calling the LLM tool-sim."""
    return Scenario(
        id="coded_0001",
        domain=Domain.BANKING,
        persona={"name": "Test"},
        user_goals=["Transfer money"],
        tools=[
            {
                "name": "initiate_transfer",
                "description": "Move funds between two accounts",
                "parameters": [
                    {"name": "from_account_id", "type": "string", "required": True},
                    {"name": "to_account_id", "type": "string", "required": True},
                    {"name": "amount", "type": "number", "required": True},
                ],
            }
        ],
        category="adaptive_tool_use",
        initial_message="Move $500 from BUS-CHK-001 to BUS-SAV-001.",
        ground_truth={
            "accounts": {
                "BUS-CHK-001": {"balance": 1000.0},
                "BUS-SAV-001": {"balance": 0.0},
            },
            "transfers_executed": [],
        },
        expected_state_changes=[
            {"assert": "accounts.BUS-SAV-001.balance", "op": "increased_by", "value": 500.0},
        ],
    )


_TRANSFER_ARGS = {
    "from_account_id": "BUS-CHK-001",
    "to_account_id": "BUS-SAV-001",
    "amount": 500.0,
}


class TestCodedTransitionWiring:
    """Phase 1b: a registered coded transition is the sole authority over the
    world mutation; the LLM tool-sim is bypassed entirely for that tool."""

    def test_coded_transition_mutates_world_and_bypasses_llm(self, monkeypatch):
        agent = ScriptedAgent(
            responses=[
                _ai_with_tool_call("initiate_transfer", _TRANSFER_ARGS, "call_1"),
                _ai_text("Transfer complete."),
            ]
        )
        # Tool-sim returns GARBAGE. If the coded path is taken the garbage is
        # never read, so the world still mutates correctly and no parse failure
        # is counted. If the LLM path were (wrongly) taken, the garbage would
        # fail to parse, the balance would not move, and parse_failures would be 1.
        runner, _ = _make_runner([], tool_text="garbage not json")
        _patch_create_model(monkeypatch, agent)

        result = runner.run(_coded_transfer_scenario(), SPEC)

        assert result.final_world["accounts"]["BUS-CHK-001"]["balance"] == 500.0
        assert result.final_world["accounts"]["BUS-SAV-001"]["balance"] == 500.0
        # The #102 mirror-write survives (it is in scope: no tool declares writes).
        assert len(result.final_world["transfers_executed"]) == 1
        assert result.tool_sim_parse_failures == 0
        assert result.coded_transition_calls == 1
        assert result.llm_tool_sim_calls == 0

    def test_agent_sees_coded_response_not_delta(self, monkeypatch):
        agent = ScriptedAgent(
            responses=[
                _ai_with_tool_call("initiate_transfer", _TRANSFER_ARGS, "call_1"),
                _ai_text("Done."),
            ]
        )
        runner, _ = _make_runner([], tool_text="garbage not json")
        _patch_create_model(monkeypatch, agent)

        result = runner.run(_coded_transfer_scenario(), SPEC)
        tool_turn = next(t for t in result.turns if t.role == "tool")
        assert "state_delta" not in tool_turn.content
        # The coded transition's contract-shaped response reaches the agent.
        assert "completed" in tool_turn.content

    def test_coded_transition_is_deterministic_across_runs(self, monkeypatch):
        # Two runs with an unstable (garbage) tool-sim still produce a
        # byte-identical graded world — the determinism #87 exists to guarantee.
        worlds = []
        for _ in range(2):
            agent = ScriptedAgent(
                responses=[
                    _ai_with_tool_call("initiate_transfer", _TRANSFER_ARGS, "call_1"),
                    _ai_text("Done."),
                ]
            )
            runner, _ = _make_runner([], tool_text="garbage not json")
            _patch_create_model(monkeypatch, agent)
            worlds.append(runner.run(_coded_transfer_scenario(), SPEC).final_world)
        import json as _json

        assert _json.dumps(worlds[0], sort_keys=True) == _json.dumps(worlds[1], sort_keys=True)

    def test_coded_in_task_error_leaves_world_untouched(self, monkeypatch):
        # Insufficient funds: the coded transition returns an error with an empty
        # delta. The world is untouched, the agent reads the error, and NO parse
        # failure is counted (the coded path never parses LLM output).
        over_budget = {**_TRANSFER_ARGS, "amount": 999999.0}
        agent = ScriptedAgent(
            responses=[
                _ai_with_tool_call("initiate_transfer", over_budget, "call_1"),
                _ai_text("Sorry, that failed."),
            ]
        )
        runner, _ = _make_runner([], tool_text="garbage not json")
        _patch_create_model(monkeypatch, agent)

        result = runner.run(_coded_transfer_scenario(), SPEC)
        assert result.final_world["accounts"]["BUS-CHK-001"]["balance"] == 1000.0
        assert result.final_world["transfers_executed"] == []
        assert result.tool_sim_parse_failures == 0
        assert result.coded_transition_calls == 1
        tool_turn = next(t for t in result.turns if t.role == "tool")
        assert "error" in tool_turn.content.lower()

    def test_unregistered_tool_falls_back_to_llm(self, monkeypatch):
        # The scenario's tool ("transfer") has no coded transition, so the run
        # falls back to the LLM tool-sim — counters reflect the split.
        agent = ScriptedAgent(
            responses=[
                _ai_with_tool_call("transfer", {"amount": 500}, "call_1"),
                _ai_text("Done."),
            ]
        )
        delta_json = (
            '{"response": {"status": "ok"}, "state_delta": {"accounts.SAV.balance": 500.0}}'
        )
        runner, _ = _make_runner([], tool_text=delta_json)
        _patch_create_model(monkeypatch, agent)

        result = runner.run(_stateful_scenario(), SPEC)
        assert result.final_world["accounts"]["SAV"]["balance"] == 500.0
        assert result.coded_transition_calls == 0
        assert result.llm_tool_sim_calls == 1

    def test_read_only_coded_transition_leaves_world_unchanged(self, monkeypatch):
        # get_account_balance is a registered READ transition: empty delta, world
        # unchanged, contract-shaped response, coded counter incremented.
        scenario = _coded_transfer_scenario()
        scenario.tools = [
            {
                "name": "get_account_balance",
                "description": "Get an account balance",
                "parameters": [{"name": "account_id", "type": "string", "required": True}],
            }
        ]
        scenario.initial_message = "What is my balance?"
        agent = ScriptedAgent(
            responses=[
                _ai_with_tool_call("get_account_balance", {"account_id": "BUS-CHK-001"}, "call_1"),
                _ai_text("Your balance is $1000."),
            ]
        )
        runner, _ = _make_runner([], tool_text="garbage not json")
        _patch_create_model(monkeypatch, agent)

        result = runner.run(scenario, SPEC)
        assert result.final_world["accounts"]["BUS-CHK-001"]["balance"] == 1000.0
        assert result.coded_transition_calls == 1
        assert result.llm_tool_sim_calls == 0
        tool_turn = next(t for t in result.turns if t.role == "tool")
        assert "current_balance" in tool_turn.content


# --------------------------------------------------------------------------- #
# Phase 3: the spine trusts ONLY coded mutations (issue #87)
# --------------------------------------------------------------------------- #
class TestSpineTrustGuard:
    """An LLM-fallback tool call that MUTATES the graded world is counted so the
    state grade can be marked non-gradable; coded mutations and read-only
    fallbacks are not counted."""

    def test_llm_fallback_mutation_is_counted(self, monkeypatch):
        # "transfer" is unregistered -> LLM fallback. The sim authors a mutating
        # delta, so llm_tool_sim_mutations is 1 (and the world did change).
        agent = ScriptedAgent(
            responses=[
                _ai_with_tool_call("transfer", {"amount": 500}, "call_1"),
                _ai_text("Done."),
            ]
        )
        delta_json = (
            '{"response": {"status": "ok"}, "state_delta": {"accounts.SAV.balance": 500.0}}'
        )
        runner, _ = _make_runner([], tool_text=delta_json)
        _patch_create_model(monkeypatch, agent)

        result = runner.run(_stateful_scenario(), SPEC)
        assert result.llm_tool_sim_mutations == 1
        assert result.llm_tool_sim_calls == 1

    def test_llm_fallback_read_only_is_not_counted(self, monkeypatch):
        # Unregistered tool, but the sim returns an EMPTY delta (a lookup). It
        # mutated nothing, so it does not taint the graded world.
        agent = ScriptedAgent(
            responses=[
                _ai_with_tool_call("transfer", {"amount": 0}, "call_1"),
                _ai_text("Nothing to do."),
            ]
        )
        delta_json = '{"response": {"status": "noop"}, "state_delta": {}}'
        runner, _ = _make_runner([], tool_text=delta_json)
        _patch_create_model(monkeypatch, agent)

        result = runner.run(_stateful_scenario(), SPEC)
        assert result.llm_tool_sim_mutations == 0
        assert result.llm_tool_sim_calls == 1

    def test_coded_mutation_is_not_counted_as_llm(self, monkeypatch):
        # A registered coded transition mutates deterministically -> it is NOT an
        # LLM mutation, so the spine-trust counter stays 0 and the grade is trusted.
        agent = ScriptedAgent(
            responses=[
                _ai_with_tool_call("initiate_transfer", _TRANSFER_ARGS, "call_1"),
                _ai_text("Done."),
            ]
        )
        runner, _ = _make_runner([], tool_text="garbage not json")
        _patch_create_model(monkeypatch, agent)

        result = runner.run(_coded_transfer_scenario(), SPEC)
        assert result.coded_transition_calls == 1
        assert result.llm_tool_sim_mutations == 0


class TestCounterPartitioningAndGuards:
    """A run mixing coded + LLM-fallback calls partitions the counters correctly;
    the clamp interacts with counting as documented; and a malformed coded return
    fails loud instead of silently mis-grading."""

    def test_mixed_run_partitions_counters(self, monkeypatch):
        # One coded call (initiate_transfer) + one unregistered call (legacy_note)
        # that the LLM sim mutates. The counters must split 1/1 with exactly one
        # LLM-authored mutation, and BOTH mutations land in the world.
        scenario = Scenario(
            id="mixed_0001",
            domain=Domain.BANKING,
            persona={"name": "Test"},
            user_goals=["Transfer and note"],
            tools=[
                {"name": "initiate_transfer", "description": "x", "parameters": []},
                {"name": "legacy_note", "description": "x", "parameters": []},
            ],
            category="adaptive_tool_use",
            initial_message="Transfer then note.",
            ground_truth={
                "accounts": {
                    "BUS-CHK-001": {"balance": 1000.0},
                    "BUS-SAV-001": {"balance": 0.0},
                },
                "transfers_executed": [],
                "notes": [],
            },
        )
        agent = ScriptedAgent(
            responses=[
                _ai_with_tool_call("initiate_transfer", _TRANSFER_ARGS, "call_1"),
                _ai_with_tool_call("legacy_note", {"text": "hi"}, "call_2"),
                _ai_text("Both done."),
            ]
        )
        # The sim text is used ONLY for the unregistered legacy_note call.
        note_delta = (
            '{"response": {"ok": true}, "state_delta": {"notes": {"__append__": {"t": 1}}}}'
        )
        runner, _ = _make_runner([], tool_text=note_delta)
        _patch_create_model(monkeypatch, agent)

        result = runner.run(scenario, SPEC)
        assert result.coded_transition_calls == 1
        assert result.llm_tool_sim_calls == 1
        assert result.llm_tool_sim_mutations == 1
        # Both authorities mutated the one world.
        assert result.final_world["accounts"]["BUS-CHK-001"]["balance"] == 500.0
        assert result.final_world["notes"] == [{"t": 1}]

    def test_coded_calls_accumulate(self, monkeypatch):
        # Two coded transfers in one run -> coded_transition_calls == 2, no LLM
        # mutation (proves accumulation, not a per-call reset).
        agent = ScriptedAgent(
            responses=[
                _ai_with_tool_call("initiate_transfer", _TRANSFER_ARGS, "call_1"),
                _ai_with_tool_call("initiate_transfer", _TRANSFER_ARGS, "call_2"),
                _ai_text("Both done."),
            ]
        )
        runner, _ = _make_runner([], tool_text="garbage not json")
        _patch_create_model(monkeypatch, agent)

        result = runner.run(_coded_transfer_scenario(), SPEC)
        assert result.coded_transition_calls == 2
        assert result.llm_tool_sim_mutations == 0
        # 500 + 500 moved out of a 1000 balance -> 0 left.
        assert result.final_world["accounts"]["BUS-CHK-001"]["balance"] == 0.0

    def test_llm_delta_fully_clamped_is_not_counted_as_mutation(self, monkeypatch):
        # An unregistered tool that declares a `writes` allow-list; the LLM sim
        # writes ONLY an out-of-scope key. The clamp empties the delta BEFORE the
        # count, so nothing is applied and llm_tool_sim_mutations stays 0 — the
        # "LLM tried to mutate but was fully clamped -> world untainted" boundary.
        scenario = Scenario(
            id="clamp_0001",
            domain=Domain.BANKING,
            persona={"name": "Test"},
            user_goals=["x"],
            tools=[
                {
                    "name": "scoped_tool",
                    "description": "x",
                    "parameters": [],
                    "writes": ["allowed"],
                }
            ],
            category="adaptive_tool_use",
            initial_message="go",
            ground_truth={"allowed": {}, "forbidden": {"key": 0}},
        )
        agent = ScriptedAgent(
            responses=[
                _ai_with_tool_call("scoped_tool", {}, "call_1"),
                _ai_text("done"),
            ]
        )
        out_of_scope = '{"response": {"ok": true}, "state_delta": {"forbidden.key": 99}}'
        runner, _ = _make_runner([], tool_text=out_of_scope)
        _patch_create_model(monkeypatch, agent)

        result = runner.run(scenario, SPEC)
        assert result.llm_tool_sim_calls == 1
        assert result.llm_tool_sim_mutations == 0
        # The out-of-scope write was dropped, not applied.
        assert result.final_world["forbidden"]["key"] == 0

    def test_customer_success_tool_routes_to_coded(self, monkeypatch):
        # CS-domain routing through the runner: a registered CS tool is served by
        # its coded transition (no LLM call), proving routing is domain-agnostic.
        scenario = Scenario(
            id="cs_0001",
            domain=Domain.CUSTOMER_SUCCESS,
            persona={"name": "Test"},
            user_goals=["Look up account"],
            tools=[{"name": "get_account", "description": "x", "parameters": []}],
            category="adaptive_tool_use",
            initial_message="Look up Acme.",
            ground_truth={"account": {"id": "ACC-1", "name": "Acme"}},
        )
        agent = ScriptedAgent(
            responses=[
                _ai_with_tool_call("get_account", {"query": "Acme"}, "call_1"),
                _ai_text("Found it."),
            ]
        )
        runner, _ = _make_runner([], tool_text="garbage not json")
        _patch_create_model(monkeypatch, agent)

        result = runner.run(scenario, SPEC)
        assert result.coded_transition_calls == 1
        assert result.llm_tool_sim_calls == 0
        tool_turn = next(t for t in result.turns if t.role == "tool")
        assert "Acme" in tool_turn.content

    def test_malformed_coded_return_raises(self, monkeypatch):
        # Defense-in-depth: a coded transition that returns a dict MISSING
        # 'response' would otherwise silently leak / mis-grade. The runner must
        # fail loud (the spine trusts coded returns completely, so a programmer
        # error there cannot be allowed to degrade quietly).
        monkeypatch.setattr(
            "eval.simulation.runner.get_transition",
            lambda domain, tool: lambda args, world: {"state_delta": {}},
        )
        agent = ScriptedAgent(
            responses=[
                _ai_with_tool_call("initiate_transfer", _TRANSFER_ARGS, "call_1"),
                _ai_text("done"),
            ]
        )
        runner, _ = _make_runner([], tool_text="garbage not json")
        _patch_create_model(monkeypatch, agent)

        with pytest.raises(ValueError, match="malformed"):
            runner.run(_coded_transfer_scenario(), SPEC)


# --------------------------------------------------------------------------- #
# Legacy (no ground_truth) path is unchanged: stateless, final_world is None
# --------------------------------------------------------------------------- #
class TestLegacyStatelessPath:
    def test_legacy_final_world_none_and_raw_tool_result(self, monkeypatch):
        agent = ScriptedAgent(
            responses=[
                _ai_with_tool_call("get_balance", {"account_id": "A1"}, "call_1"),
                _ai_text("Your balance is $500."),
            ]
        )
        # Stateless sim returns a plain JSON blob; it's fed back verbatim.
        runner, _ = _make_runner([], tool_text='{"balance": 1234.56}')
        _patch_create_model(monkeypatch, agent)

        result = runner.run(_scenario(), SPEC)  # _scenario() has no ground_truth

        assert result.final_world is None
        tool_turn = next(t for t in result.turns if t.role == "tool")
        assert tool_turn.content == '{"balance": 1234.56}'


class TestCompletionDecoupling:
    """User-sim completion is decoupled from goal-completion (#32, part 1).

    The user sim only signals it is done talking; whether the goals are met is
    the deterministic state check at the moment of ending. A sim that ends early
    with unmet state assertions is flagged premature, NOT silently passed.
    """

    def test_sim_complete_with_unmet_assertions_is_premature(self, monkeypatch):
        # Agent calls transfer once; the tool sim returns an EMPTY state_delta,
        # so SAV.balance never moves and the increased_by assertion stays unmet.
        # The user sim says COMPLETE anyway -> premature ending.
        agent = ScriptedAgent(
            responses=[
                _ai_with_tool_call("transfer", {"amount": 500}, "call_1"),
                _ai_text("All set!"),
            ]
        )
        no_op_delta = '{"response": {"status": "ok"}, "state_delta": {}}'
        runner, _ = _make_runner([], tool_text=no_op_delta, user_text=CONVERSATION_COMPLETE)
        _patch_create_model(monkeypatch, agent)

        result = runner.run(_stateful_scenario(), SPEC)

        assert result.completed is True  # sim ended the conversation
        assert result.ended_by == "user_sim"
        # SAV.balance unchanged (0.0), so the increased_by 500 assertion fails:
        # state progress is 0/1 = 0.0, below 1.0 -> premature.
        assert result.state_progress_at_end == 0.0
        assert result.premature_end is True

    def test_sim_complete_with_all_assertions_met_is_clean(self, monkeypatch):
        # Tool sim moves SAV.balance up by exactly 500, satisfying the assertion.
        agent = ScriptedAgent(
            responses=[
                _ai_with_tool_call("transfer", {"amount": 500}, "call_1"),
                _ai_text("Done, $500 moved."),
            ]
        )
        good_delta = (
            '{"response": {"status": "ok"}, "state_delta": {"accounts.SAV.balance": 500.0}}'
        )
        runner, _ = _make_runner([], tool_text=good_delta, user_text=CONVERSATION_COMPLETE)
        _patch_create_model(monkeypatch, agent)

        result = runner.run(_stateful_scenario(), SPEC)

        assert result.completed is True
        assert result.ended_by == "user_sim"
        # 1/1 assertions pass -> full progress, not premature.
        assert result.state_progress_at_end == 1.0
        assert result.premature_end is False

    def test_max_turns_exhaustion_is_distinct(self, monkeypatch):
        # The user sim NEVER says COMPLETE (it keeps replying), so the loop runs
        # out the turn budget. That is ended_by == "max_turns", not user_sim,
        # and never premature regardless of state progress.
        # Agent always produces a tool-free message so each user turn is one
        # round; the user sim keeps the conversation going.
        agent = ScriptedAgent(responses=[_ai_text("How else can I help?")])
        no_op_delta = '{"response": {"status": "ok"}, "state_delta": {}}'
        runner, _ = _make_runner([], tool_text=no_op_delta, user_text="I still need more help.")
        _patch_create_model(monkeypatch, agent)

        result = runner.run(_stateful_scenario(), SPEC)

        assert result.completed is False  # sim never declared done
        assert result.ended_by == "max_turns"
        assert result.premature_end is False
        # State progress is still recorded for max-turns runs (0.0 here: SAV
        # never moved), so a timed-out run is comparable to a sim-ended one.
        assert result.state_progress_at_end == 0.0

    def test_legacy_scenario_has_no_state_progress(self, monkeypatch):
        # A scenario with no ground_truth has no deterministic check to gate on,
        # so state_progress_at_end is None and premature_end is False even when
        # the sim ends the conversation.
        agent = ScriptedAgent(
            responses=[
                _ai_with_tool_call("get_balance", {"account_id": "A1"}, "call_1"),
                _ai_text("Your balance is $500."),
            ]
        )
        runner, _ = _make_runner([], tool_text='{"balance": 500}', user_text=CONVERSATION_COMPLETE)
        _patch_create_model(monkeypatch, agent)

        result = runner.run(_scenario(), SPEC)  # _scenario() has no ground_truth

        assert result.completed is True
        assert result.ended_by == "user_sim"
        assert result.state_progress_at_end is None
        assert result.premature_end is False


class TestUserKnownFacts:
    """The user simulator must know its own identity facts (smoke-run finding:
    without them it invents verification values and every identity gate fails
    for harness reasons)."""

    def _scenario(self, gt):
        from eval.config import Domain
        from eval.simulation.runner import Scenario

        return Scenario(
            id="t",
            domain=Domain.BANKING,
            persona={"name": "T"},
            user_goals=["g"],
            tools=[],
            category="adaptive_tool_use",
            initial_message="hi",
            ground_truth=gt,
        )

    def test_banking_shape(self):
        from eval.simulation.runner import SimulationRunner

        gt = {
            "customer": {"customer_id": "CUST-1", "ssn_last4": "4417", "verified": False},
            "accounts": {"BUS-CHK-001": {"type": "checking", "balance": 100.0}},
        }
        known = SimulationRunner._user_known_facts(self._scenario(gt))
        assert known["customer"]["ssn_last4"] == "4417"
        assert "verified" not in known["customer"]  # server-side state stays hidden
        assert known["your_accounts"] == {"BUS-CHK-001": "checking"}  # ids+types, no balances

    def test_cs_shape(self):
        from eval.simulation.runner import SimulationRunner

        gt = {"account": {"account_id": "ACCT-9", "company_name": "Acme", "health_score": 58}}
        known = SimulationRunner._user_known_facts(self._scenario(gt))
        assert known["your_account"] == {"account_id": "ACCT-9", "company_name": "Acme"}

    def test_legacy_scenario_returns_none(self):
        from eval.simulation.runner import SimulationRunner

        assert SimulationRunner._user_known_facts(self._scenario(None)) is None
