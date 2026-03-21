"""Multi-turn agent simulation runner.

Orchestrates conversations between a user simulator, the agent under test,
and a tool simulator. Tracks tokens, latency, and full conversation history
for downstream scoring.
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from eval.config import DEFAULT_SIMULATION, DOMAIN_CONFIGS, Domain, SimulationConfig
from eval.providers.registry import ModelSpec, create_model

logger = logging.getLogger(__name__)

CONVERSATION_COMPLETE = "[CONVERSATION_COMPLETE]"

# Matches JSON tool_call objects, handling nested braces in arguments
_TOOL_CALL_RE = re.compile(
    r'\{\s*"tool_call"\s*:\s*(\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\})\s*\}'
)


@dataclass
class ToolCall:
    """A single tool call made by the agent."""

    turn: int
    tool_name: str
    arguments: dict
    result: str


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""

    turn_number: int
    role: str  # "user", "agent", "tool"
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    latency_ms: float = 0.0
    token_count: int = 0


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


@dataclass
class Scenario:
    """A test scenario to run."""

    id: str
    domain: Domain
    persona: dict
    user_goals: list[str]
    tools: list[dict]  # JSON schema tool definitions
    category: str
    initial_message: str


class SimulationRunner:
    """Runs multi-turn agent simulations."""

    def __init__(self, config: SimulationConfig | None = None):
        self.config = config or DEFAULT_SIMULATION

        # User simulator — generates realistic user turns
        self._user_sim = create_model(
            ModelSpec(
                name="user_simulator",
                model_id=self.config.user_simulator_model,
                provider="openai",
                temperature=self.config.user_simulator_temperature,
            )
        )

        # Tool simulator — generates realistic tool responses
        self._tool_sim = create_model(
            ModelSpec(
                name="tool_simulator",
                model_id=self.config.tool_simulator_model,
                provider="openai",
                temperature=self.config.tool_simulator_temperature,
            )
        )

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

        turns: list[ConversationTurn] = []
        total_input_tokens = 0
        total_output_tokens = 0
        total_latency_ms = 0.0
        completed = False

        # Build agent system message with tools
        tools_json = json.dumps(scenario.tools, indent=2)
        agent_system = (
            f"{domain_config['system_prompt']}\n\n"
            f"Available tools:\n{tools_json}\n\n"
            "When you need to use a tool, respond with a JSON tool call in this format:\n"
            '{"tool_call": {"name": "<tool_name>", "arguments": {<args>}}}\n\n'
            "You may make multiple tool calls in sequence. After receiving tool results, "
            "provide your response to the user."
        )

        # Conversation history for the agent
        agent_messages = [SystemMessage(content=agent_system)]

        # Start with user's initial message
        current_user_message = scenario.initial_message

        for turn_num in range(self.config.max_turns):
            # Record user turn
            turns.append(
                ConversationTurn(
                    turn_number=turn_num,
                    role="user",
                    content=current_user_message,
                )
            )
            agent_messages.append(HumanMessage(content=current_user_message))

            # Agent turn
            start = time.perf_counter()
            try:
                agent_response = agent.invoke(agent_messages)
            except Exception as e:
                logger.error("Agent error on turn %d: %s", turn_num, e)
                return SimulationResult(
                    scenario_id=scenario.id,
                    domain=scenario.domain.value,
                    model_name=agent_spec.name,
                    turns=turns,
                    total_turns=turn_num,
                    total_latency_ms=total_latency_ms,
                    total_input_tokens=total_input_tokens,
                    total_output_tokens=total_output_tokens,
                    completed=False,
                    error=str(e),
                )
            agent_latency = (time.perf_counter() - start) * 1000

            agent_content = (
                agent_response.content
                if isinstance(agent_response.content, str)
                else str(agent_response.content)
            )
            agent_messages.append(AIMessage(content=agent_content))

            # Track tokens from response metadata
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
            total_latency_ms += agent_latency

            # Check for tool calls in agent response
            tool_calls_this_turn = self._extract_tool_calls(
                agent_content, turn_num
            )

            # Simulate tool responses if agent made tool calls
            for tc in tool_calls_this_turn:
                tool_result = self._simulate_tool(tc, scenario.tools)
                tc.result = tool_result
                # Feed tool result back to agent
                tool_msg = f"Tool result for {tc.tool_name}:\n{tool_result}"
                agent_messages.append(HumanMessage(content=tool_msg))
                turns.append(
                    ConversationTurn(
                        turn_number=turn_num,
                        role="tool",
                        content=tool_msg,
                    )
                )

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

            # Generate next user turn (or detect completion)
            user_response = self._simulate_user_turn(
                scenario, turns, agent_content
            )

            if CONVERSATION_COMPLETE in user_response:
                completed = True
                break

            current_user_message = user_response

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
        )

    def _extract_tool_calls(self, content: str, turn: int) -> list[ToolCall]:
        """Extract tool calls from agent response content."""
        calls = []

        # First try: whole response is a single tool call JSON
        try:
            parsed = json.loads(content)
            if "tool_call" in parsed:
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

        # Second try: find embedded tool_call JSON objects (handles nested braces)
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
        self, tool_call: ToolCall, available_tools: list[dict]
    ) -> str:
        """Use an LLM to generate a realistic tool response."""
        tool_schema = next(
            (t for t in available_tools if t.get("name") == tool_call.tool_name),
            None,
        )

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
        return (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )

    def _simulate_user_turn(
        self,
        scenario: Scenario,
        history: list[ConversationTurn],
        last_agent_response: str,
    ) -> str:
        """Generate the next user message based on persona and goals."""
        # Last 10 turns for context window efficiency
        history_text = "\n".join(
            f"[{t.role}] {t.content}" for t in history[-10:]
        )

        prompt = (
            "You are simulating a user in a conversation with an AI agent.\n\n"
            f"Persona: {json.dumps(scenario.persona)}\n"
            "Goals (pursue these naturally across the conversation):\n"
        )
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

        response = self._user_sim.invoke([HumanMessage(content=prompt)])
        return (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )
