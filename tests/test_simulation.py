"""Tests for simulation runner — tool call extraction and data structures."""

from eval.simulation.runner import (
    _TOOL_CALL_RE,
    ConversationTurn,
    ToolCall,
)


class TestToolCallExtraction:
    """Test the tool call regex and extraction logic."""

    def test_simple_tool_call(self):
        content = '{"tool_call": {"name": "get_balance", "arguments": {"account_id": "123"}}}'
        matches = _TOOL_CALL_RE.findall(content)
        assert len(matches) == 1

    def test_nested_arguments(self):
        content = (
            '{"tool_call": {"name": "create_transfer", '
            '"arguments": {"from": {"id": "A1", "type": "checking"}, "amount": 500}}}'
        )
        matches = _TOOL_CALL_RE.findall(content)
        assert len(matches) == 1

    def test_tool_call_in_text(self):
        content = (
            "I'll check your balance now.\n"
            '{"tool_call": {"name": "get_balance", "arguments": {"id": "X"}}}\n'
            "Let me look that up."
        )
        matches = _TOOL_CALL_RE.findall(content)
        assert len(matches) == 1

    def test_no_tool_call(self):
        content = "I can help you with that. Let me explain the process."
        matches = _TOOL_CALL_RE.findall(content)
        assert len(matches) == 0

    def test_multiple_tool_calls(self):
        content = (
            '{"tool_call": {"name": "tool_a", "arguments": {}}}\n'
            '{"tool_call": {"name": "tool_b", "arguments": {"x": 1}}}'
        )
        matches = _TOOL_CALL_RE.findall(content)
        assert len(matches) == 2


class TestDataStructures:
    def test_tool_call_fields(self):
        tc = ToolCall(turn=0, tool_name="test", arguments={"a": 1}, result="ok")
        assert tc.turn == 0
        assert tc.tool_name == "test"

    def test_conversation_turn_defaults(self):
        turn = ConversationTurn(turn_number=0, role="user", content="hi")
        assert turn.tool_calls == []
        assert turn.latency_ms == 0.0
        assert turn.token_count == 0
