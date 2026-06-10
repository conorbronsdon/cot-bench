"""OpenInference trace emission for COT Bench.

Emits traces in OpenInference format so results can be ingested by
Arize Phoenix or any OTel-compatible backend.

Two exporters run side by side:

- An :class:`InMemorySpanExporter` so tests can introspect emitted spans via
  :func:`get_collected_spans`.
- A dependency-free :class:`JSONLFileSpanExporter` that, when a trace directory
  is configured, writes every finished span to ``spans.jsonl`` (one JSON object
  per line). These are real, durable traces — not in-memory-only — so a run's
  spans survive the process and can be loaded into Arize Phoenix (via its
  OTLP/file import) or any tool that reads OpenInference-attributed spans.
"""

import json
import os
import threading
from pathlib import Path

# OpenInference semantic conventions
from openinference.semconv.trace import (
    SpanAttributes,
)
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

_provider: TracerProvider | None = None
_exporter: InMemorySpanExporter | None = None
_file_exporter: "JSONLFileSpanExporter | None" = None


class JSONLFileSpanExporter(SpanExporter):
    """Write finished spans as JSON lines to a file.

    Dependency-free: uses only the OTel SDK's in-SDK ``SpanExporter`` primitive
    plus ``json`` from the stdlib. Each span becomes one line in ``spans.jsonl``
    capturing the OpenInference-attributed name, attributes, and timestamps —
    enough to reconstruct the trace in Phoenix or any OTel/OpenInference reader.
    """

    def __init__(self, output_path: str | Path):
        self._path = Path(output_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @staticmethod
    def _span_to_dict(span: ReadableSpan) -> dict:
        ctx = span.get_span_context()
        return {
            "name": span.name,
            "trace_id": format(ctx.trace_id, "032x") if ctx else None,
            "span_id": format(ctx.span_id, "016x") if ctx else None,
            "parent_id": (format(span.parent.span_id, "016x") if span.parent else None),
            "kind": str(span.kind),
            "start_time": span.start_time,
            "end_time": span.end_time,
            "status": str(span.status.status_code),
            "attributes": dict(span.attributes or {}),
        }

    def export(self, spans) -> SpanExportResult:
        lines = [json.dumps(self._span_to_dict(s)) for s in spans]
        with self._lock, open(self._path, "a", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True


def init_tracing(
    service_name: str = "cot-bench",
    trace_dir: str | Path | None = None,
) -> TracerProvider:
    """Initialize OpenTelemetry tracing with OpenInference conventions.

    Args:
        service_name: ``service.name`` resource attribute.
        trace_dir: Directory to write ``spans.jsonl`` into. When ``None``,
            falls back to the ``COT_BENCH_TRACE_DIR`` env var; if neither is
            set, only the in-memory exporter is attached (test/default-off
            behavior). When a directory IS resolved, a real on-disk JSONL
            exporter is attached alongside the in-memory one.
    """
    global _provider, _exporter, _file_exporter

    from opentelemetry.sdk.resources import Resource

    resource = Resource.create({"service.name": service_name})
    _exporter = InMemorySpanExporter()
    _provider = TracerProvider(resource=resource)
    _provider.add_span_processor(SimpleSpanProcessor(_exporter))

    resolved_dir = trace_dir or os.environ.get("COT_BENCH_TRACE_DIR")
    if resolved_dir:
        spans_path = Path(resolved_dir) / "spans.jsonl"
        _file_exporter = JSONLFileSpanExporter(spans_path)
        _provider.add_span_processor(SimpleSpanProcessor(_file_exporter))
    else:
        _file_exporter = None

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
