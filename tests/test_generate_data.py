"""Tests for the multi-author generation guard (no API calls)."""

import pytest

from scripts.generate_data import (
    AUTHOR_MODELS,
    assert_author_allowed,
    resolve_author,
)


class TestAuthorGuard:
    def test_model_under_test_id_rejected(self):
        # gpt-4.1 is a contestant in MODELS_UNDER_TEST.
        with pytest.raises(RuntimeError, match="MODELS_UNDER_TEST"):
            assert_author_allowed("gpt-4.1")

    def test_model_under_test_display_name_rejected(self):
        with pytest.raises(RuntimeError, match="MODELS_UNDER_TEST"):
            assert_author_allowed("Claude Sonnet 4.6")

    def test_clean_author_allowed(self):
        # A non-contestant author id (no prefix relation to any contestant)
        # passes the guard.
        assert_author_allowed("anthropic/claude-opus-4.6")  # should not raise

    def test_resolve_friendly_name(self):
        # claude-opus is the default friendly author (OpenRouter-served).
        spec = resolve_author("claude-opus")
        assert spec["model_id"] == AUTHOR_MODELS["claude-opus"]["model_id"]
        assert spec["provider"] == "openrouter"

    def test_resolve_raw_model_id_defaults_openai(self):
        spec = resolve_author("some-clean-author-x")
        assert spec["model_id"] == "some-clean-author-x"
        assert spec["provider"] == "openai"

    def test_resolve_rejects_contestant_friendly_and_raw(self):
        # Default friendly author gpt-4.1 maps to a contestant -> guarded.
        with pytest.raises(RuntimeError, match="MODELS_UNDER_TEST"):
            resolve_author("gpt-4.1")


class TestAuthorGuardFamilyMatching:
    """Pinned snapshots and bare family ids must both be blocked (prefix match)."""

    def test_bare_family_id_blocked_when_contestant_is_pinned(self):
        # Contestants are pinned to dated snapshots (gpt-4.1-2025-04-14);
        # the bare family id is still the same model and must be rejected.
        with pytest.raises(RuntimeError, match="MODELS_UNDER_TEST"):
            assert_author_allowed("gpt-4.1")

    def test_dated_snapshot_of_contestant_blocked(self):
        with pytest.raises(RuntimeError, match="MODELS_UNDER_TEST"):
            assert_author_allowed("gpt-4.1-2025-04-14")

    def test_unrelated_family_allowed(self):
        # gpt-4.5-preview shares no prefix relation with any contestant id.
        assert_author_allowed("gpt-4.5-preview")
