"""Tests for the JSONL file span exporter.

These pin the honest-trace fix: when COT_BENCH_TRACE_DIR is set, finished spans
are written to spans.jsonl on disk (not held only in memory and discarded).
A judge-evaluation span must land as a JSONL line carrying the expected
OpenInference attributes.

No network: spans are emitted directly via the tracer.
"""

import json

from openinference.semconv.trace import SpanAttributes

import eval.tracing as tracing
from eval.tracing import get_tracer, init_tracing, trace_judge_evaluation


def _reset_tracer_provider():
    """OTel set_tracer_provider warns + ignores re-sets within a process.

    Clear the override flag so each test installs a fresh provider with the
    exporter it wants.
    """
    from opentelemetry import trace as _trace

    _trace._TRACER_PROVIDER = None
    _trace._TRACER_PROVIDER_SET_ONCE._done = False


def test_judge_span_written_to_jsonl(tmp_path, monkeypatch):
    _reset_tracer_provider()
    monkeypatch.setenv("COT_BENCH_TRACE_DIR", str(tmp_path))

    init_tracing()
    tracer = get_tracer()

    trace_judge_evaluation(
        tracer,
        judge_name="Kimi K2.6",
        scenario_id="banking_001",
        rubric_type="task_completion",
        score=0.85,
        reasoning="Goal fully met",
        latency_ms=12.3,
    )

    # Flush the SimpleSpanProcessor pipeline to disk.
    tracing._provider.force_flush()

    spans_file = tmp_path / "spans.jsonl"
    assert spans_file.exists()

    lines = [ln for ln in spans_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 1
    rec = json.loads(lines[0])

    assert rec["name"] == "judge_evaluation"
    attrs = rec["attributes"]
    assert attrs[SpanAttributes.OPENINFERENCE_SPAN_KIND] == "EVALUATOR"
    assert attrs[SpanAttributes.LLM_MODEL_NAME] == "Kimi K2.6"
    assert attrs["cot_bench.scenario_id"] == "banking_001"
    assert attrs["cot_bench.rubric_type"] == "task_completion"
    assert attrs["cot_bench.score"] == 0.85
    assert attrs["cot_bench.reasoning"] == "Goal fully met"
    # Timestamps survive as integers.
    assert isinstance(rec["start_time"], int)
    assert isinstance(rec["end_time"], int)


def test_no_trace_dir_means_no_file(tmp_path, monkeypatch):
    # Without COT_BENCH_TRACE_DIR (and no explicit dir), only the in-memory
    # exporter is attached — nothing is written to disk.
    _reset_tracer_provider()
    monkeypatch.delenv("COT_BENCH_TRACE_DIR", raising=False)

    init_tracing()
    tracer = get_tracer()
    trace_judge_evaluation(tracer, "Opus", "s1", "tool_selection", 0.5, "ok", 1.0)
    tracing._provider.force_flush()

    assert not (tmp_path / "spans.jsonl").exists()
    # In-memory spans are still collectible.
    assert len(tracing.get_collected_spans()) == 1


def test_explicit_trace_dir_arg(tmp_path, monkeypatch):
    # The trace_dir argument works without the env var.
    _reset_tracer_provider()
    monkeypatch.delenv("COT_BENCH_TRACE_DIR", raising=False)

    init_tracing(trace_dir=tmp_path / "nested")
    tracer = get_tracer()
    trace_judge_evaluation(tracer, "GLM", "s2", "task_completion", 0.6, "fine", 2.0)
    tracing._provider.force_flush()

    spans_file = tmp_path / "nested" / "spans.jsonl"
    assert spans_file.exists()
    rec = json.loads(spans_file.read_text(encoding="utf-8").strip())
    assert rec["attributes"]["cot_bench.score"] == 0.6
