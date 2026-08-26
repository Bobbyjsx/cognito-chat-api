"""Tests for ProviderRegistry central resolution."""

import pytest

from app.models.config import AppConfigDB, ModelProvider
from app.providers.base import BaseProvider, ProviderModelNotFoundError, ProviderUnsupportedError
from app.providers.claude import ClaudeProvider
from app.providers.gemini import GeminiProvider
from app.providers.registry import ProviderRegistry, create_default_provider_registry


class DummyProvider(BaseProvider):
    name = "dummy"

    async def generate(self, model, contents, config=None):
        pass

    async def generate_stream(self, model, contents, config=None):
        pass

    def build_tools(self, tool_configs):
        return []

    async def parts_for_attachment(self, attachment, data):
        return []

    async def transcribe_audio(self, model, audio_bytes, mime_type, prompt):
        return "", 0


def test_registry_register_and_get():
    registry = ProviderRegistry()
    dummy = DummyProvider()

    registry.register("dummy", dummy)
    registry.register(ModelProvider.OPENAI, dummy)

    assert registry.get("dummy") is dummy
    assert registry.get("DUMMY") is dummy
    assert registry.get(ModelProvider.OPENAI) is dummy


def test_registry_get_unregistered_raises():
    registry = ProviderRegistry()
    with pytest.raises(ProviderUnsupportedError):
        registry.get("nonexistent")


def test_registry_get_for_model_from_config():
    registry = ProviderRegistry()
    gemini = GeminiProvider(api_key="x")
    claude = ClaudeProvider(api_key="y")

    registry.register(ModelProvider.GOOGLE, gemini)
    registry.register(ModelProvider.ANTHROPIC, claude)

    config = AppConfigDB()

    # Gemini model
    prov_gemini = registry.get_for_model("gemini-3.6-flash", config=config)
    assert prov_gemini is gemini

    # Claude model
    prov_claude = registry.get_for_model("claude-3-7-sonnet", config=config)
    assert prov_claude is claude


def test_registry_get_for_model_prefix_fallback():
    registry = ProviderRegistry()
    gemini = GeminiProvider(api_key="x")
    claude = ClaudeProvider(api_key="y")

    registry.register("gemini", gemini)
    registry.register("anthropic", claude)

    # Prefix fallback without config entry
    assert registry.get_for_model("claude-next-gen") is claude
    assert registry.get_for_model("gemini-experimental-x") is gemini


def test_registry_unknown_model_raises_model_not_found():
    registry = ProviderRegistry()
    with pytest.raises(ProviderModelNotFoundError):
        registry.get_for_model("unsupported-vendor-model-99")


def test_create_default_provider_registry():
    registry = create_default_provider_registry()
    assert "gemini" in registry.list_providers()
    assert "anthropic" in registry.list_providers()
    assert registry.supports_model("gemini-3.6-flash")
    assert registry.supports_model("claude-3-7-sonnet")
