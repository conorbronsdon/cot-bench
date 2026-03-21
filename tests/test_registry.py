"""Tests for the model provider registry."""


from eval.providers.registry import _REGISTRY, ModelSpec, register_provider


class TestRegistry:
    def test_all_providers_registered(self):
        expected = [
            "openai", "anthropic", "google", "max",
            "deepseek", "qwen", "together", "mistral",
        ]
        for provider in expected:
            assert provider in _REGISTRY, f"Provider '{provider}' not registered"

    def test_register_custom_provider(self):
        def _dummy_factory(spec):
            return f"dummy-{spec.model_id}"

        register_provider("test_provider", _dummy_factory)
        assert "test_provider" in _REGISTRY

        # Cleanup
        del _REGISTRY["test_provider"]

    def test_model_spec_defaults(self):
        spec = ModelSpec(name="test", model_id="test-model", provider="openai")
        assert spec.temperature == 0.0
        assert spec.max_tokens == 4096
        assert spec.endpoint is None
