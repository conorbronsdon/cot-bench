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
        # gpt-4.5-preview is on neither the under-test nor judge lists.
        assert_author_allowed("gpt-4.5-preview")  # should not raise

    def test_resolve_friendly_name(self):
        spec = resolve_author("gpt-4.5")
        assert spec["model_id"] == AUTHOR_MODELS["gpt-4.5"]["model_id"]
        assert spec["provider"] == "openai"

    def test_resolve_raw_model_id_defaults_openai(self):
        spec = resolve_author("some-clean-author-x")
        assert spec["model_id"] == "some-clean-author-x"
        assert spec["provider"] == "openai"

    def test_resolve_rejects_contestant_friendly_and_raw(self):
        # Default friendly author gpt-4.1 maps to a contestant -> guarded.
        with pytest.raises(RuntimeError, match="MODELS_UNDER_TEST"):
            resolve_author("gpt-4.1")
