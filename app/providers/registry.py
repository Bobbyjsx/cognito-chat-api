"""Central Provider Registry for multi-model resolution.

The registry is the single source of truth for resolving concrete AI providers
(Gemini, Claude, OpenAI) from model names and provider identifiers.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.models.config import AppConfigDB, ModelProvider
from app.providers.base import BaseProvider, ProviderModelNotFoundError, ProviderUnsupportedError

if TYPE_CHECKING:
    from app.core.config import Settings

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """Registry managing instantiated AI model providers."""

    def __init__(self):
        self._providers: dict[str, BaseProvider] = {}

    def register(self, key: str | ModelProvider, provider: BaseProvider) -> None:
        """Register a provider instance under one or more string/enum keys."""
        normalized_key = key.value.lower() if isinstance(key, ModelProvider) else str(key).lower()
        self._providers[normalized_key] = provider
        logger.debug("Registered provider '%s' under key '%s'", provider.name, normalized_key)

    def get(self, key: str | ModelProvider) -> BaseProvider:
        """Retrieve a registered provider by key or provider enum."""
        normalized_key = key.value.lower() if isinstance(key, ModelProvider) else str(key).lower()
        provider = self._providers.get(normalized_key)
        if provider is None:
            raise ProviderUnsupportedError(
                f"Provider '{normalized_key}' is not registered. Available: {list(self._providers.keys())}"
            )
        return provider

    def get_for_model(self, model: str, config: AppConfigDB | None = None) -> BaseProvider:
        """Resolve the appropriate provider for a given model ID.

        1. Looks up the model in the model registry (AppConfigDB.models_list) to read its declared provider.
        2. If not found in config, falls back to standard model name prefixes (e.g. 'claude-' -> anthropic, 'gemini-' -> google).
        """
        if not model or model.lower() in ("auto", "smart", "default", "none"):
            # For auto/smart routing, fallback to default provider
            if "google" in self._providers:
                return self._providers["google"]
            if "gemini" in self._providers:
                return self._providers["gemini"]

        if config is not None and model in config.models_list:
            model_cfg = config.models_list[model]
            provider_key = model_cfg.provider.value.lower()
            if provider_key in self._providers:
                return self._providers[provider_key]
            # Try common aliases
            if provider_key == "google" and "gemini" in self._providers:
                return self._providers["gemini"]
            if provider_key == "anthropic" and "claude" in self._providers:
                return self._providers["claude"]

        # Prefix-based resolution fallback
        lower_model = model.lower()
        if lower_model.startswith("claude") or "anthropic" in lower_model:
            for key in ("anthropic", "claude"):
                if key in self._providers:
                    return self._providers[key]

        if (
            lower_model.startswith("gemini")
            or "imagen" in lower_model
            or "veo" in lower_model
            or "google" in lower_model
        ):
            for key in ("gemini", "google"):
                if key in self._providers:
                    return self._providers[key]

        if lower_model.startswith(("gpt", "o1", "o3")) or "openai" in lower_model:
            for key in ("openai", "gpt"):
                if key in self._providers:
                    return self._providers[key]

        # Check if any provider explicitly claims support for this model
        for provider in self._providers.values():
            if provider.supports_model(model):
                return provider

        raise ProviderModelNotFoundError(f"No provider registered that supports model '{model}'.")

    def supports_model(self, model: str, config: AppConfigDB | None = None) -> bool:
        """Check if any registered provider can serve the given model."""
        try:
            self.get_for_model(model, config)
            return True
        except ProviderModelNotFoundError:
            return False

    def list_providers(self) -> list[str]:
        """List unique names of all registered providers."""
        return list({p.name for p in self._providers.values()})


def create_default_provider_registry(settings: Settings | None = None) -> ProviderRegistry:
    """Factory creating a ProviderRegistry pre-populated with Gemini and Claude providers."""
    if settings is None:
        from app.core.config import settings as global_settings

        settings = global_settings

    from app.providers.claude import ClaudeProvider
    from app.providers.gemini import GeminiProvider

    registry = ProviderRegistry()

    # Register Gemini provider
    gemini_provider = GeminiProvider(api_key=settings.gemini_api_key)
    registry.register("gemini", gemini_provider)
    registry.register("google", gemini_provider)
    registry.register(ModelProvider.GOOGLE, gemini_provider)

    # Register Claude provider via AWS Bedrock
    import os

    bedrock_key = (
        getattr(settings, "aws_bedrock_api_key", "")
        or getattr(settings, "aws_bearer_token_bedrock", "")
        or os.getenv("AWS_BEARER_TOKEN_BEDROCK", "")
        or os.getenv("AWS_BEDROCK_API_KEY", "")
        or os.getenv("AWS_API_KEY", "")
        or getattr(settings, "anthropic_api_key", "")
    )
    bedrock_region = getattr(settings, "aws_region", "eu-west-1") or os.getenv("AWS_REGION", "eu-west-1")
    bedrock_access_key = getattr(settings, "aws_access_key_id", "") or os.getenv("AWS_ACCESS_KEY_ID", "")
    bedrock_secret_key = getattr(settings, "aws_secret_access_key", "") or os.getenv("AWS_SECRET_ACCESS_KEY", "")
    bedrock_session_token = getattr(settings, "aws_session_token", "") or os.getenv("AWS_SESSION_TOKEN", "")

    claude_provider = ClaudeProvider(
        api_key=bedrock_key,
        aws_region=bedrock_region,
        aws_access_key_id=bedrock_access_key,
        aws_secret_access_key=bedrock_secret_key,
        aws_session_token=bedrock_session_token,
    )
    registry.register("anthropic", claude_provider)
    registry.register("claude", claude_provider)
    registry.register("bedrock", claude_provider)
    registry.register("bedrock_claude", claude_provider)
    registry.register("aws_bedrock", claude_provider)
    registry.register(ModelProvider.ANTHROPIC, claude_provider)

    return registry
