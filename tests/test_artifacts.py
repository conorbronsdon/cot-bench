"""Tests for per-run artifact persistence and the JSONL trace exporter.

The leaderboard's promise is auditability: every published score must trace
back to the conversation and raw judge outputs it came from. These tests pin
the artifact file layout, JSON content, slug rules, the --no-artifacts opt-out,
and that judge spans are actually written to disk as JSONL.

No network: faked SimulationResult/ConsensusResult/JudgeResult objects only.
"""

import json

from eval.artifacts import build_artifact, model_slug, write_run_artifact
from eval.scoring.judge import ConsensusResult, JudgeResult
from eval.simulation.runner import ConversationTurn, SimulationResult, ToolCall


def _sim_result():
    """A SimulationResult with a user -> agent(tool) -> tool -> agent flow."""
    turns = [
        ConversationTurn(turn_number=0, role="user", content="Move $500 to savings"),
        ConversationTurn(
            turn_number=0,
            role="agent",
            content="Let me transfer that for you.",
            tool_calls=[
                ToolCall(
                    turn=0,
                    tool_name="transfer_funds",
                    arguments={"amount": 500, "to": "savings"},
                    result="",
                    tool_call_id="call_0_1",
                )
            ],
            latency_ms=42.0,
            token_count=120,
        ),
        ConversationTurn(
            turn_number=0,
            role="tool",
            content='{"status": "ok"}',
            tool_call_id="call_0_1",
        ),
        ConversationTurn(
            turn_number=0,
            role="agent",
            content="Done — $500 moved to savings.",
            latency_ms=31.0,
            token_count=40,
        ),
    ]
    return SimulationResult(
        scenario_id="banking_001",
        domain="banking",
        model_name="GPT-4.1",
        turns=turns,
        total_turns=2,
        total_latency_ms=73.0,
        total_input_tokens=100,
        total_output_tokens=60,
        completed=True,
        error=None,
    )


def _judge(name, score, rubric_type, parse_failed=False, raw=None):
    return JudgeResult(
        judge_name=name,
        rubric_type=rubric_type,
        overall_score=score,
        reasoning=f"{name} reasoning",
        raw_response=raw if raw is not None else {"overall_score": score},
        latency_ms=5.0,
        parse_failed=parse_failed,
    )


def _consensus(rubric_type, judge_results):
    return ConsensusResult(
        scenario_id="banking_001",
        rubric_type=rubric_type,
        judge_results=judge_results,
        consensus_score=0.8,
        agreement_rate=1.0,
        max_disagreement=0.1,
        n_judges_requested=2,
        n_judges_valid=len([j for j in judge_results if not j.parse_failed]),
    )


class TestModelSlug:
    def test_dotted_version(self):
        assert model_slug("GPT-4.1") == "gpt-4-1"

    def test_spaces_and_dots(self):
        assert model_slug("Claude Sonnet 4.6") == "claude-sonnet-4-6"

    def test_collapses_runs_and_strips(self):
        assert model_slug("  Llama 4 -- Maverick  ") == "llama-4-maverick"

    def test_already_clean(self):
        assert model_slug("deepseek") == "deepseek"


class TestBuildArtifact:
    def test_payload_shape(self):
        tc = _consensus(
            "task_completion",
            [_judge("Kimi", 0.8, "task_completion"), _judge("Opus", 0.9, "task_completion")],
        )
        ts = _consensus("tool_selection", [_judge("Kimi", 0.7, "tool_selection")])
        art = build_artifact("banking_001", "GPT-4.1", 2, _sim_result(), tc, ts)

        assert art["scenario_id"] == "banking_001"
        assert art["model"] == "GPT-4.1"
        assert art["run_index"] == 2
        assert "evaluated_at" in art

        # Transcript preserves all turns incl. tool calls/results/ids.
        assert len(art["transcript"]) == 4
        agent_turn = art["transcript"][1]
        assert agent_turn["role"] == "agent"
        assert agent_turn["tool_calls"][0]["tool_name"] == "transfer_funds"
        assert agent_turn["tool_calls"][0]["arguments"] == {"amount": 500, "to": "savings"}
        assert agent_turn["tool_calls"][0]["tool_call_id"] == "call_0_1"
        tool_turn = art["transcript"][2]
        assert tool_turn["role"] == "tool"
        assert tool_turn["tool_call_id"] == "call_0_1"

        # Per-judge outputs for both rubric types.
        assert [j["judge_name"] for j in art["judges"]["task_completion"]] == ["Kimi", "Opus"]
        assert art["judges"]["task_completion"][0]["reasoning"] == "Kimi reasoning"
        assert art["judges"]["task_completion"][0]["raw_response"] == {"overall_score": 0.8}
        assert art["judges"]["tool_selection"][0]["judge_name"] == "Kimi"

        # sim_meta.
        assert art["sim_meta"]["completed"] is True
        assert art["sim_meta"]["total_turns"] == 2
        assert art["sim_meta"]["input_tokens"] == 100
        assert art["sim_meta"]["output_tokens"] == 60
        assert art["sim_meta"]["error"] is None
        # Completion-decoupling fields (#32) are present with sane defaults.
        assert art["sim_meta"]["ended_by"] == "max_turns"
        assert art["sim_meta"]["state_progress_at_end"] is None
        assert art["sim_meta"]["premature_end"] is False

    def test_domain_category_holdout_persisted(self):
        """Issue #46/#31: domain, category, and holdout are top-level fields."""
        tc = _consensus("task_completion", [_judge("Kimi", 0.8, "task_completion")])
        ts = _consensus("tool_selection", [_judge("Kimi", 0.7, "tool_selection")])
        art = build_artifact(
            "cs_adaptive_tool_use_0001",
            "GPT-4.1",
            0,
            _sim_result(),
            tc,
            ts,
            domain="customer_success",
            category="adaptive_tool_use",
            holdout=True,
        )
        assert art["domain"] == "customer_success"
        assert art["category"] == "adaptive_tool_use"
        assert art["holdout"] is True

    def test_domain_category_default_none_holdout_false(self):
        """Backward-compatible defaults when a caller omits the new fields."""
        tc = _consensus("task_completion", [_judge("Kimi", 0.8, "task_completion")])
        ts = _consensus("tool_selection", [_judge("Kimi", 0.7, "tool_selection")])
        art = build_artifact("banking_001", "GPT-4.1", 0, _sim_result(), tc, ts)
        assert art["domain"] is None
        assert art["category"] is None
        assert art["holdout"] is False

    def test_sim_meta_carries_premature_ending(self):
        tc = _consensus("task_completion", [_judge("Kimi", 0.8, "task_completion")])
        ts = _consensus("tool_selection", [_judge("Kimi", 0.7, "tool_selection")])
        sim = _sim_result()
        # A run the user sim ended before the state check passed (#32).
        sim.ended_by = "user_sim"
        sim.state_progress_at_end = 0.5
        sim.premature_end = True
        art = build_artifact("banking_001", "GPT-4.1", 0, sim, tc, ts)
        assert art["sim_meta"]["ended_by"] == "user_sim"
        assert art["sim_meta"]["state_progress_at_end"] == 0.5
        assert art["sim_meta"]["premature_end"] is True

    def test_sim_meta_carries_dual_control(self):
        # Dual control (#58): dual_control / user_actions_fired /
        # user_actions_suppressed / coordination_ok are persisted in sim_meta so
        # the coordination_rate denominator (fired-action rows only) and the
        # trigger-met-but-undeliverable suppression (the #74 fired-but-not-
        # delivered class) are auditable from artifacts and a resume
        # reconstructs the row identically.
        tc = _consensus("task_completion", [_judge("Kimi", 0.8, "task_completion")])
        ts = _consensus("tool_selection", [_judge("Kimi", 0.7, "tool_selection")])
        sim = _sim_result()
        sim.dual_control = True
        sim.user_actions_fired = 1
        sim.user_actions_suppressed = 1
        sim.coordination_ok = True
        art = build_artifact("banking_001", "GPT-4.1", 0, sim, tc, ts)
        assert art["sim_meta"]["dual_control"] is True
        assert art["sim_meta"]["user_actions_fired"] == 1
        assert art["sim_meta"]["user_actions_suppressed"] == 1
        assert art["sim_meta"]["coordination_ok"] is True

    def test_sim_meta_dual_control_defaults_for_single_control(self):
        tc = _consensus("task_completion", [_judge("Kimi", 0.8, "task_completion")])
        ts = _consensus("tool_selection", [_judge("Kimi", 0.7, "tool_selection")])
        art = build_artifact("banking_001", "GPT-4.1", 0, _sim_result(), tc, ts)
        assert art["sim_meta"]["dual_control"] is False
        assert art["sim_meta"]["user_actions_fired"] == 0
        assert art["sim_meta"]["user_actions_suppressed"] == 0
        assert art["sim_meta"]["coordination_ok"] is None

    def test_no_state_block_when_state_omitted(self):
        tc = _consensus("task_completion", [_judge("Kimi", 0.8, "task_completion")])
        ts = _consensus("tool_selection", [_judge("Kimi", 0.7, "tool_selection")])
        art = build_artifact("banking_001", "GPT-4.1", 0, _sim_result(), tc, ts)
        assert "state" not in art

    def test_state_block_included_when_provided(self):
        tc = _consensus("task_completion", [_judge("Kimi", 0.8, "task_completion")])
        ts = _consensus("tool_selection", [_judge("Kimi", 0.7, "tool_selection")])
        sim = _sim_result()
        sim.final_world = {"accounts": {"A1": {"balance": 150.0}}}
        state = {
            "score": 1.0,
            "checks": [{"passed": True, "detail": "balance increased"}],
            "n_passed": 1,
            "n_total": 1,
        }
        art = build_artifact("banking_001", "GPT-4.1", 0, sim, tc, ts, state=state)
        assert art["state"]["score"] == 1.0
        assert art["state"]["checks"][0]["passed"] is True
        assert art["state"]["final_world"] == {"accounts": {"A1": {"balance": 150.0}}}

    def test_parse_failed_judge_kept_for_audit(self):
        tc = _consensus(
            "task_completion",
            [
                _judge("Kimi", 0.8, "task_completion"),
                _judge("Opus", 0.0, "task_completion", parse_failed=True, raw={}),
            ],
        )
        ts = _consensus("tool_selection", [_judge("Kimi", 0.7, "tool_selection")])
        art = build_artifact("banking_001", "GPT-4.1", 0, _sim_result(), tc, ts)
        judges = art["judges"]["task_completion"]
        assert len(judges) == 2  # parse-failed judge retained
        failed = next(j for j in judges if j["judge_name"] == "Opus")
        assert failed["parse_failed"] is True
        assert failed["raw_response"] == {}


class TestWriteRunArtifact:
    def test_layout_and_content(self, tmp_path):
        tc = _consensus("task_completion", [_judge("Kimi", 0.8, "task_completion")])
        ts = _consensus("tool_selection", [_judge("Kimi", 0.7, "tool_selection")])
        path = write_run_artifact(
            tmp_path / "artifacts",
            "results_20260609_120000",
            "banking_001",
            "GPT-4.1",
            3,
            _sim_result(),
            tc,
            ts,
        )
        # Layout: {root}/{run_id}/{model-slug}/{scenario}_run{idx}.json
        expected = (
            tmp_path / "artifacts" / "results_20260609_120000" / "gpt-4-1" / "banking_001_run3.json"
        )
        assert path == expected
        assert path.exists()

        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["scenario_id"] == "banking_001"
        assert data["run_index"] == 3
        assert len(data["transcript"]) == 4
        assert data["judges"]["task_completion"][0]["judge_name"] == "Kimi"


class TestNoArtifactsHonored:
    """evaluate_scenario must write nothing when artifacts_root is None."""

    def test_no_artifacts_writes_nothing(self, tmp_path, monkeypatch):
        import scripts.run_eval as run_eval

        sim = _sim_result()
        tc = _consensus("task_completion", [_judge("Kimi", 0.8, "task_completion")])
        ts = _consensus("tool_selection", [_judge("Kimi", 0.7, "tool_selection")])

        class _Runner:
            def run(self, scenario, agent_spec):
                return sim

        class _Spec:
            name = "GPT-4.1"
            model_id = "gpt-4.1"

        class _Domain:
            value = "banking"

        class _Scenario:
            id = "banking_001"
            domain = _Domain()
            category = "adaptive_tool_use"
            user_goals = ["move money"]
            tools = [{"name": "transfer_funds", "description": "x"}]
            ground_truth = None
            expected_state_changes = None

        # Stub out the judge layer entirely (no network).
        monkeypatch.setattr(
            run_eval, "score_with_all_judges", lambda *a, **k: tc if "task" in a[2] else ts
        )
        # Costs lookup tolerates an unknown model id.
        monkeypatch.setattr(run_eval, "TOKEN_COSTS", {})

        sentinel = {"called": False}
        orig = run_eval.write_run_artifact

        def spy(*a, **k):
            sentinel["called"] = True
            return orig(*a, **k)

        monkeypatch.setattr(run_eval, "write_run_artifact", spy)

        # tracer is a no-op recorder; init real tracing in-memory only.
        from eval.tracing import get_tracer, init_tracing

        init_tracing()
        tracer = get_tracer()

        run_eval.evaluate_scenario(
            _Runner(),
            _Scenario(),
            _Spec(),
            tracer,
            ["kimi"],
            run_id="results_x",
            run_index=0,
            artifacts_root=None,
        )
        assert sentinel["called"] is False

        # And the inverse: with a root, it IS written.
        run_eval.evaluate_scenario(
            _Runner(),
            _Scenario(),
            _Spec(),
            tracer,
            ["kimi"],
            run_id="results_x",
            run_index=0,
            artifacts_root=tmp_path / "artifacts",
        )
        assert sentinel["called"] is True
        written = tmp_path / "artifacts" / "results_x" / "gpt-4-1" / "banking_001_run0.json"
        assert written.exists()
