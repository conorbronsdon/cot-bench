"""Deterministic do-nothing ("null") agent for anti-gaming validation.

Motivated by the Berkeley RDI audit, which showed trivial agents can game
agentic benchmarks to near-perfect scores. A credible benchmark must score a
do-nothing agent near zero — on BOTH the LLM judges and the deterministic state
checks. This module provides that agent.

``NullAgentChatModel`` is a ``BaseChatModel`` that slots into the exact same
interface the simulation runner uses for every contestant (``invoke`` /
``bind_tools`` via LangChain), but makes **no API calls** and **never emits a
tool call**. Every invocation returns the same short, deflecting, user-facing
message. Because it never calls a tool, the stateful world is never mutated, so
the deterministic state grader scores it 0.0 on any scenario that expects a
state change (and on a no-mutation scenario it trivially "passes" the
no-unauthorized-mutation contract — the agent did nothing, which is the only
case where doing nothing is correct).

This makes the null agent:

- **Zero-cost and deterministic** — it requires no provider key and produces
  identical transcripts every run; that determinism is the whole point.
- **Well-formed** — it produces normal ``ConversationTurn`` transcripts and
  artifacts, so a judged run (when someone later pays for one) scores it exactly
  as it would any other contestant. No special-casing in the scoring path.

The null agent is wired in as a normal provider (``"null"``) so the runner
treats it like any other model. It is deliberately kept OUT of
``MODELS_UNDER_TEST`` and excluded from the published leaderboard (see
``scripts/aggregate_results.py``) so it can never appear as a ranked contestant.
"""

from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

# Canonical CLI name for the null agent. Used by scripts/run_eval.py to inject
# the spec on request and by scripts/aggregate_results.py to exclude it from the
# leaderboard. Lower-case to match the --models convention; the leaderboard
# guard matches case-insensitively.
NULL_AGENT_NAME = "null-agent"

# Provider key registered in the model registry.
NULL_AGENT_PROVIDER = "null"

# The single deflecting, do-nothing message returned for every turn. Trivial and
# non-committal by design: it makes no tool calls, looks up nothing, and resolves
# no goal — so judges score task completion / tool selection at the floor and the
# state grader (no mutation ever happened) scores any state-changing scenario 0.0.
NULL_AGENT_RESPONSE = (
    "I'm not able to help with that right now. Please contact support if you need assistance."
)


class NullAgentChatModel(BaseChatModel):
    """A do-nothing agent that makes no tool calls and gives only a trivial reply.

    Returns ``NULL_AGENT_RESPONSE`` for every invocation with empty
    ``tool_calls`` and zero token usage. ``bind_tools`` is a no-op returning
    ``self`` so the runner's native-tool-calling path works unchanged — the
    agent simply chooses never to call any of the bound tools.

    No network calls are ever made; the agent is fully deterministic.
    """

    invoke_count: int = 0

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.invoke_count += 1
        message = AIMessage(
            content=NULL_AGENT_RESPONSE,
            # No tool calls — this is the do-nothing behavior.
            tool_calls=[],
            # Zero, real usage: the null agent spends nothing.
            usage_metadata={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            response_metadata={"model_name": NULL_AGENT_NAME},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools: Any, **kwargs: Any) -> "NullAgentChatModel":  # noqa: ARG002
        # Accept and ignore tool bindings: the null agent never calls a tool.
        return self

    @property
    def _llm_type(self) -> str:
        return "null-agent"


def create_null_agent(spec: Any) -> BaseChatModel:  # noqa: ARG001
    """Provider factory for the null agent (signature matches the registry).

    The ``spec`` is ignored — the null agent has no model id, temperature, or
    endpoint to honor. It is deterministic by construction.
    """
    return NullAgentChatModel()
