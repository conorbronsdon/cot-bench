"""Tests for the simulation runner — native tool calling, agent/tool loop,
transcript ordering, schema conversion, and token accounting.

All chat models are stubbed; NO real API calls are made.
"""

from typing import Any

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
