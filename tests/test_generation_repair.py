"""Tests for the one-round generation repair loop + partial yield.

All offline: the author LLM call (`_call_author`) is mocked, so no OpenRouter /
OpenAI / Anthropic request is ever made. We assert on the repair-prompt content,
the re-validation outcome, discard-on-second-failure, partial-yield file writing
+ exit code, and authorship preservation through repair.
"""

import json
from unittest import mock

import pytest

from scripts import generate_data
from scripts.validate_scenarios import validate_scenario_dict

AUTHOR = {"model_id": "anthropic/claude-opus-4.8", "provider": "openrouter"}

# A persona/tools/goals payload that satisfies the non-state-assertion content
# rules, so the ONLY thing we vary in tests is the expected_state_changes block.
PERSONA = {
    "name": "Dana Lee",
    "age": 41,
    "occupation": "Operations Manager",
    "personality_traits": ["direct", "busy"],
    "tone": "brisk",
    "detail_level": "brief",
    "background": "Runs logistics for a mid-size firm.",
}

TOOLS = [
    {
        "name": "lookup_ticket",
        "description": "Look up a support ticket by id",
        "parameters": [{"name": "ticket_id", "type": "string", "description": "id"}],
    },
    {
        "name": "update_ticket",
        "description": "Update a support ticket status",
        "parameters": [{"name": "ticket_id", "type": "string", "description": "id"}],
    },
]

USER_GOALS = [
    "Check the status of support ticket LOGCORP-001",
    "Escalate the ticket to a manager",
    "Confirm the escalation was recorded",
]


def _base_scenario(expected_state_changes, ground_truth):
    """A scenario body as an AUTHOR would return it (no schema_version /
    authorship — those are stamped by the pipeline)."""
    return {
        "id": "placeholder",  # overwritten by the pipeline's content-hash id
        "category": "adaptive_tool_use",
        "persona": PERSONA,
        "user_goals": list(USER_GOALS),
        "tools": [
            {
                "name": "lookup_ticket",
                "description": "Look up a support ticket by id",
                "parameters": [{"name": "ticket_id", "type": "string", "description": "id"}],
            },
            {
                "name": "update_ticket",
                "description": "Update a support ticket status",
                "parameters": [{"name": "ticket_id", "type": "string", "description": "id"}],
            },
        ],
        "initial_message": "I need help with one of our support tickets, it's urgent.",
        "difficulty": "medium",
        "ground_truth": ground_truth,
        "expected_state_changes": expected_state_changes,
    }


# A BAD scenario: 'support_tickets' is a DICT, and the path uses [0] index syntax
# that does not resolve — exactly the two failure classes from issue #36.
BAD_GROUND_TRUTH = {
    "support_tickets": {"LOGCORP-001": {"status": "open"}},
}
BAD_STATE_CHANGES = [
    {
        "assert": "support_tickets.LOGCORP-001[0].status",
        "op": "equals",
        "value": "escalated",
        "goal": "Check the status of support ticket LOGCORP-001",
    },
    {
        "assert": "support_tickets",
        "op": "contains",
        "match": {"id": "LOGCORP-001"},
        "goal": "Escalate the ticket to a manager",
    },
]

# A GOOD scenario: 'support_tickets' is a LIST, paths resolve, contains targets a
# list — passes validation.
GOOD_GROUND_TRUTH = {
    "support_tickets": [{"id": "LOGCORP-001", "status": "open"}],
    "escalations": [],
}
GOOD_STATE_CHANGES = [
    {
        "assert": "support_tickets",
        "op": "contains",
        "match": {"id": "LOGCORP-001"},
        "goal": "Check the status of support ticket LOGCORP-001",
    },
    {
        "assert": "escalations",
        "op": "contains",
        "match": {"ticket_id": "LOGCORP-001"},
        "goal": "Escalate the ticket to a manager",
    },
]


def _bad_json():
    return json.dumps(_base_scenario(BAD_STATE_CHANGES, BAD_GROUND_TRUTH))


def _good_json():
    return json.dumps(_base_scenario(GOOD_STATE_CHANGES, GOOD_GROUND_TRUTH))


def _gen(stats=None, side_effect=None):
    """Run generate_scenario with _call_author mocked; return (result, mock)."""
    with mock.patch.object(generate_data, "_call_author", side_effect=side_effect) as m:
        result = generate_data.generate_scenario(
            domain="customer_success",
            category="adaptive_tool_use",
            persona=PERSONA,
            tools=TOOLS,
            index=0,
            author=AUTHOR,
            author_run="2026-06-10-claude-opus-batch",
            stats=stats,
        )
    return result, m


class TestSanityFixtures:
    def test_bad_fixture_actually_fails_validation(self):
        body = _base_scenario(BAD_STATE_CHANGES, BAD_GROUND_TRUTH)
        body["schema_version"] = "0.2"
        body["authorship"] = {"author_model": AUTHOR["model_id"]}
        errors = validate_scenario_dict(body)
        assert any("does not resolve" in e for e in errors)
        assert any("list" in e for e in errors)

    def test_good_fixture_passes_validation(self):
        body = _base_scenario(GOOD_STATE_CHANGES, GOOD_GROUND_TRUTH)
        body["schema_version"] = "0.2"
        body["authorship"] = {"author_model": AUTHOR["model_id"]}
        assert validate_scenario_dict(body) == []


class TestRepairLoop:
    def test_first_pass_valid_no_repair(self):
        stats = {}
        result, m = _gen(stats=stats, side_effect=[_good_json()])
        assert result is not None
        assert m.call_count == 1  # no repair call
        assert stats == {"attempted": 1}

    def test_repair_invoked_with_error_text_and_accepted(self):
        # First call returns a BAD scenario, repair call returns a GOOD one.
        stats = {}
        result, m = _gen(stats=stats, side_effect=[_bad_json(), _good_json()])
        assert result is not None
        assert m.call_count == 2  # generation + one repair

        # The repair call's messages must contain the EXACT validator error text.
        repair_args = m.call_args_list[1]
        messages = (
            repair_args.args[1] if len(repair_args.args) > 1 else repair_args.kwargs["messages"]
        )
        repair_text = messages[-1]["content"]
        assert "does not resolve in ground_truth" in repair_text
        assert "requires" in repair_text and "list" in repair_text
        # The failing scenario JSON is echoed back for correction.
        assert "support_tickets" in repair_text
        assert stats["repaired"] == 1
        assert stats.get("discarded", 0) == 0

    def test_second_failure_discarded(self):
        # Both passes return a BAD scenario -> discard, no result.
        stats = {}
        result, m = _gen(stats=stats, side_effect=[_bad_json(), _bad_json()])
        assert result is None
        assert m.call_count == 2
        assert stats["discarded"] == 1
        assert stats.get("repaired", 0) == 0

    def test_repair_unparseable_json_discarded(self):
        stats = {}
        result, m = _gen(stats=stats, side_effect=[_bad_json(), "not json at all {{{"])
        assert result is None
        assert stats["discarded"] == 1

    def test_repair_empty_completion_discarded(self):
        stats = {}
        result, m = _gen(stats=stats, side_effect=[_bad_json(), None])
        assert result is None
        assert stats["discarded"] == 1


class TestAuthorshipPreservedThroughRepair:
    def test_author_unchanged_and_repaired_flag_set(self):
        result, _ = _gen(side_effect=[_bad_json(), _good_json()])
        assert result is not None
        assert result["authorship"]["author_model"] == AUTHOR["model_id"]
        assert result["authorship"]["author_run"] == "2026-06-10-claude-opus-batch"
        assert result["authorship"]["repaired"] is True

    def test_no_repair_flag_false(self):
        result, _ = _gen(side_effect=[_good_json()])
        assert result is not None
        assert result["authorship"]["author_model"] == AUTHOR["model_id"]
        assert result["authorship"]["repaired"] is False

    def test_repair_cannot_change_author_to_contestant(self):
        # Even if the repaired body tried to inject a contestant author, the
        # pipeline re-stamps authorship from the configured author. We simulate a
        # repaired body that smuggles in an authorship block.
        smuggled = json.loads(_good_json())
        smuggled["authorship"] = {"author_model": "gpt-4.1"}
        result, _ = _gen(side_effect=[_bad_json(), json.dumps(smuggled)])
        assert result is not None
        # Re-stamp wins: author stays the clean configured author.
        assert result["authorship"]["author_model"] == AUTHOR["model_id"]


class TestThreadSafeCounters:
    def test_counters_increment(self):
        c = generate_data._Counters()
        c["attempted"] = c.get("attempted", 0) + 1
        c["repaired"] = c.get("repaired", 0) + 1
        d = c.as_dict()
        assert d["attempted"] == 1
        assert d["repaired"] == 1
        assert d["discarded"] == 0


class TestPartialYield:
    def _run_main(self, tmp_path, side_effects, monkeypatch):
        """Drive main() end-to-end with mocked author + filesystem in tmp_path."""
        # Pre-seed tools.json + personas.json so main() skips generating them
        # (those paths would otherwise need their own mocked author calls).
        domain = "customer_success"
        out = tmp_path / "data"
        (out / "domains" / domain).mkdir(parents=True)
        (out / "domains" / domain / "tools.json").write_text(json.dumps(TOOLS))
        (out / "domains" / domain / "personas.json").write_text(json.dumps([PERSONA]))

        argv = [
            "generate_data",
            "--domain",
            domain,
            "--author-model",
            "claude-opus",
            "--scenarios-per-category",
            "1",
            "--categories",
            "adaptive_tool_use",
            "--output-dir",
            str(out),
            "--max-workers",
            "1",
        ]
        monkeypatch.setattr("sys.argv", argv)

        scenario_dir = out / "scenarios" / domain
        with mock.patch.object(generate_data, "_call_author", side_effect=side_effects):
            with pytest.raises(SystemExit) as exc:
                generate_data.main()
        written = sorted(scenario_dir.glob("*.json")) if scenario_dir.exists() else []
        return exc.value.code, written

    def test_partial_yield_writes_valid_exits_zero(self, tmp_path, monkeypatch):
        # Two categories' worth would need 2 scenarios; here one category, one
        # scenario, BAD then GOOD on repair -> one valid file, exit 0.
        code, written = self._run_main(tmp_path, [_bad_json(), _good_json()], monkeypatch)
        assert code == 0
        assert len(written) == 1
        data = json.loads(written[0].read_text())
        assert data["authorship"]["repaired"] is True
        assert validate_scenario_dict(data) == []

    def test_total_failure_exits_nonzero(self, tmp_path, monkeypatch):
        # The only scenario fails both passes -> nothing written -> exit nonzero.
        code, written = self._run_main(tmp_path, [_bad_json(), _bad_json()], monkeypatch)
        assert code == 1
        assert written == []

    def test_step_summary_written_when_env_set(self, tmp_path, monkeypatch):
        summary_file = tmp_path / "summary.md"
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_file))
        code, written = self._run_main(tmp_path, [_good_json()], monkeypatch)
        assert code == 0
        text = summary_file.read_text(encoding="utf-8")
        assert "attempted" in text
        assert "repaired" in text
        assert "discarded" in text
