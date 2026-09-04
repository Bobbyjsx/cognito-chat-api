"""Provider abstraction layer.

The application talks to AI providers exclusively through the interfaces in
``app.providers.base`` and resolves them via ``ProviderRegistry``. SDK-specific
logic is encapsulated inside concrete providers under this package.
"""

from app.providers.base import (
    BaseProvider,
    ContentPart,
    GenerationConfig,
    GenerationEvent,
    GenerationResult,
    ProviderAuthError,
    ProviderConnectionError,
    ProviderError,
    ProviderGenerationError,
    ProviderInvalidRequestError,
    ProviderModelNotFoundError,
    ProviderOverloadedError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnsupportedError,
    ToolCall,
    ToolResult,
    classify_provider_error,
)
from app.providers.claude import ClaudeProvider
from app.providers.gemini import GeminiProvider
from app.providers.registry import ProviderRegistry, create_default_provider_registry

__all__ = [
    "BaseProvider",
    "ClaudeProvider",
    "ContentPart",
    "GeminiProvider",
    "GenerationConfig",
    "GenerationEvent",
    "GenerationResult",
    "ProviderAuthError",
    "ProviderConnectionError",
    "ProviderError",
    "ProviderGenerationError",
    "ProviderInvalidRequestError",
    "ProviderModelNotFoundError",
    "ProviderOverloadedError",
    "ProviderRateLimitError",
    "ProviderRegistry",
    "ProviderTimeoutError",
    "ProviderUnsupportedError",
    "ToolCall",
    "ToolResult",
    "classify_provider_error",
    "create_default_provider_registry",
]
