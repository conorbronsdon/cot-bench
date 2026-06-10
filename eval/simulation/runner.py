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

import json
import logging
import re
import time
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from eval.config import DEFAULT_SIMULATION, DOMAIN_CONFIGS, Domain, SimulationConfig
from eval.providers.registry import ModelSpec, create_model

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
        total_latency_ms = 0.0
        completed = False

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
                    )
                agent_latency = (time.perf_counter() - start) * 1000
                total_latency_ms += agent_latency

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
                    tool_result = self._simulate_tool(tc, scenario.tools)
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

            # Generate next user turn (or detect completion).
            user_response = self._simulate_user_turn(scenario, turns, last_agent_content)
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

    def _simulate_tool(self, tool_call: ToolCall, available_tools: list[dict]) -> str:
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
        return response.content if isinstance(response.content, str) else str(response.content)

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
        return response.content if isinstance(response.content, str) else str(response.content)
