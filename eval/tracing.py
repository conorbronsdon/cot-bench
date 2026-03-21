"""OpenInference trace emission for COT Bench.

Emits traces in OpenInference format so results can be ingested by
Arize Phoenix or any OTel-compatible backend.
"""

# OpenInference semantic conventions
from openinference.semconv.trace import (
    SpanAttributes,
)
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory import InMemorySpanExporter

_provider: TracerProvider | None = None
_exporter: InMemorySpanExporter | None = None


def init_tracing(service_name: str = "cot-bench") -> TracerProvider:
    """Initialize OpenTelemetry tracing with OpenInference conventions."""
    global _provider, _exporter

    _exporter = InMemorySpanExporter()
    _provider = TracerProvider()
    _provider.add_span_processor(SimpleSpanProcessor(_exporter))
    trace.set_tracer_provider(_provider)

    return _provider


def get_tracer(name: str = "cot-bench.eval") -> trace.Tracer:
    """Get a tracer instance."""
    if _provider is None:
        init_tracing()
    return trace.get_tracer(name)


def trace_agent_turn(
    tracer: trace.Tracer,
    model_name: str,
    input_text: str,
    output_text: str,
    tool_calls: list[dict] | None = None,
    token_count_input: int = 0,
    token_count_output: int = 0,
    latency_ms: float = 0.0,
):
    """Record an agent turn as an OpenInference span.

    Args:
        tracer: The OTel tracer to use.
        model_name: Name of the model under test.
        input_text: The user/system input to the agent.
        output_text: The agent's response.
        tool_calls: List of tool calls made (if any).
        token_count_input: Input token count.
        token_count_output: Output token count.
        latency_ms: Response latency in milliseconds.
    """
    with tracer.start_as_current_span("agent_turn") as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "AGENT")
        span.set_attribute(SpanAttributes.INPUT_VALUE, input_text)
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, output_text)
        span.set_attribute(SpanAttributes.LLM_MODEL_NAME, model_name)
        span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, token_count_input)
        span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, token_count_output)
        span.set_attribute("cot_bench.latency_ms", latency_ms)

        if tool_calls:
            for i, tc in enumerate(tool_calls):
                span.set_attribute(f"cot_bench.tool_calls.{i}.name", tc.get("name", ""))
                span.set_attribute(
                    f"cot_bench.tool_calls.{i}.arguments",
                    str(tc.get("arguments", {})),
                )


def trace_judge_evaluation(
    tracer: trace.Tracer,
    judge_name: str,
    scenario_id: str,
    rubric_type: str,
    score: float,
    reasoning: str,
    latency_ms: float,
):
    """Record a judge evaluation as an OpenInference span."""
    with tracer.start_as_current_span("judge_evaluation") as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "EVALUATOR")
        span.set_attribute(SpanAttributes.LLM_MODEL_NAME, judge_name)
        span.set_attribute("cot_bench.scenario_id", scenario_id)
        span.set_attribute("cot_bench.rubric_type", rubric_type)
        span.set_attribute("cot_bench.score", score)
        span.set_attribute("cot_bench.reasoning", reasoning)
        span.set_attribute("cot_bench.latency_ms", latency_ms)


def get_collected_spans() -> list:
    """Return all collected spans (for export/debugging)."""
    if _exporter is None:
        return []
    return _exporter.get_finished_spans()
