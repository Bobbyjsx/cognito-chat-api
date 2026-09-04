"""Provider-agnostic interfaces and value types.

The rest of the application should depend on these abstractions (and on
``GenerationConfig`` / ``ContentPart``) instead of importing SDK classes
directly. Only concrete providers may import provider SDKs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal

# ────────────────────────────────────────────────────────────────────────────
# Errors
# ────────────────────────────────────────────────────────────────────────────

SAFE_GENERATION_ERROR = "Model generation failed. Please try again."

ERROR_CODE_MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
ERROR_CODE_GENERATION_FAILED = "GENERATION_FAILED"
ERROR_CODE_TOOL_FAILED = "TOOL_EXECUTION_FAILED"
ERROR_CODE_RATE_LIMIT = "RATE_LIMIT_EXCEEDED"
ERROR_CODE_AUTH_FAILED = "AUTHENTICATION_FAILED"
ERROR_CODE_INVALID_REQUEST = "INVALID_REQUEST"
ERROR_CODE_OVERLOADED = "PROVIDER_OVERLOADED"
ERROR_CODE_TIMEOUT = "TIMEOUT"
ERROR_CODE_CONNECTION = "CONNECTION_FAILED"
ERROR_CODE_UNSUPPORTED = "UNSUPPORTED_OPERATION"


class ProviderError(Exception):
    """Base class for all provider failures surfaced to callers."""


class ProviderModelNotFoundError(ProviderError):
    """The requested model does not exist on the provider."""


class ProviderGenerationError(ProviderError):
    """Generation failed with a provider-side error."""

    def __init__(self, message: str = SAFE_GENERATION_ERROR, status_code: int = 500):
        super().__init__(message)
        self.status_code = status_code


class ProviderRateLimitError(ProviderGenerationError):
    """Rate limit / quota exceeded on the provider API."""

    def __init__(self, message: str = "Rate limit exceeded. Please try again later.", status_code: int = 429):
        super().__init__(message=message, status_code=status_code)


class ProviderAuthError(ProviderGenerationError):
    """Authentication or authorization failure with provider API."""

    def __init__(self, message: str = "Provider authentication failed.", status_code: int = 401):
        super().__init__(message=message, status_code=status_code)


class ProviderInvalidRequestError(ProviderGenerationError):
    """Malformed request or invalid parameters sent to provider."""

    def __init__(self, message: str = "Invalid request to provider.", status_code: int = 400):
        super().__init__(message=message, status_code=status_code)


class ProviderOverloadedError(ProviderGenerationError):
    """Provider API is overloaded or temporarily unavailable."""

    def __init__(
        self, message: str = "Provider is currently overloaded. Please try again shortly.", status_code: int = 503
    ):
        super().__init__(message=message, status_code=status_code)


class ProviderTimeoutError(ProviderGenerationError):
    """Request to provider timed out."""

    def __init__(self, message: str = "Provider request timed out.", status_code: int = 504):
        super().__init__(message=message, status_code=status_code)


class ProviderConnectionError(ProviderGenerationError):
    """Failed to establish connection to provider API."""

    def __init__(self, message: str = "Failed to connect to provider.", status_code: int = 502):
        super().__init__(message=message, status_code=status_code)


class ProviderUnsupportedError(ProviderGenerationError):
    """Requested model capability or feature is not supported by provider."""

    def __init__(self, message: str = "Operation not supported by provider.", status_code: int = 400):
        super().__init__(message=message, status_code=status_code)


def classify_provider_error(exc: Exception) -> tuple[int, str, str]:
    """Return ``(status_code, error_code, message)`` for a raised exception.

    Mirrors the behaviour of the historical ``extract_genai_error`` helper so
    HTTP semantics stay stable, while keeping SDK-specific exceptions inside
    the provider layer.
    """
    if isinstance(exc, ProviderModelNotFoundError):
        return 404, ERROR_CODE_MODEL_NOT_FOUND, str(exc) or SAFE_GENERATION_ERROR
    if isinstance(exc, ProviderRateLimitError):
        return exc.status_code, ERROR_CODE_RATE_LIMIT, str(exc) or "Rate limit exceeded."
    if isinstance(exc, ProviderAuthError):
        return exc.status_code, ERROR_CODE_AUTH_FAILED, str(exc) or "Authentication failed."
    if isinstance(exc, ProviderInvalidRequestError):
        return exc.status_code, ERROR_CODE_INVALID_REQUEST, str(exc) or "Invalid request."
    if isinstance(exc, ProviderOverloadedError):
        return exc.status_code, ERROR_CODE_OVERLOADED, str(exc) or "Provider overloaded."
    if isinstance(exc, ProviderTimeoutError):
        return exc.status_code, ERROR_CODE_TIMEOUT, str(exc) or "Request timed out."
    if isinstance(exc, ProviderConnectionError):
        return exc.status_code, ERROR_CODE_CONNECTION, str(exc) or "Connection error."
    if isinstance(exc, ProviderUnsupportedError):
        return exc.status_code, ERROR_CODE_UNSUPPORTED, str(exc) or "Unsupported operation."
    if isinstance(exc, ProviderGenerationError):
        return exc.status_code, ERROR_CODE_GENERATION_FAILED, str(exc) or SAFE_GENERATION_ERROR
    if isinstance(exc, ProviderError):
        return 500, ERROR_CODE_GENERATION_FAILED, str(exc) or SAFE_GENERATION_ERROR
    return 500, ERROR_CODE_GENERATION_FAILED, SAFE_GENERATION_ERROR


# ────────────────────────────────────────────────────────────────────────────
# Value types (provider-agnostic)
# ────────────────────────────────────────────────────────────────────────────

TOOL_KIND_FUNCTION = "function"
TOOL_KIND_SERVER = "server"


@dataclass
class ToolCall:
    """A tool invocation requested by the model."""

    id: str
    name: str
    args: dict[str, Any]
    kind: str = TOOL_KIND_FUNCTION  # "function" → executed by the app, "server" → handled by the provider


@dataclass
class ToolResult:
    """The outcome of a tool invocation."""

    id: str
    name: str
    output: Any
    is_error: bool = False
    kind: str = TOOL_KIND_FUNCTION


@dataclass
class GenerationEvent:
    """One event in a streaming generation.

    ``type`` values: ``text``, ``reasoning``, ``tool_call``, ``tool_result``,
    ``usage``, ``error``.
    """

    type: Literal["text", "reasoning", "tool_call", "tool_result", "usage", "error"]
    token: str | None = None
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None
    total_tokens: int | None = None


@dataclass
class GenerationResult:
    """A complete non-streaming generation."""

    text: str
    total_tokens: int
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class GenerationConfig:
    """Provider-agnostic generation configuration.

    ``tool_configs`` holds internal tool definitions produced by the tool
    registry; the provider converts them into native tool objects.
    """

    system_instruction: str | None = None
    thinking_budget: int | None = None
    include_thoughts: bool = False
    tool_configs: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ContentPart:
    """A provider-agnostic content turn.

    ``parts`` is a list of plain dicts with one of the following shapes:
    ``{"text": str}``, ``{"inline_data": {"mime_type", "data" (base64)}}``,
    ``{"file_data": {"file_uri", "mime_type"}}``,
    ``{"function_call": {"id", "name", "args"}}`` or
    ``{"function_response": {"id", "name", "response"}}``.
    """

    role: str  # "user" | "model"
    parts: list[dict[str, Any]] = field(default_factory=list)


# ────────────────────────────────────────────────────────────────────────────
# Provider interface
# ────────────────────────────────────────────────────────────────────────────


class BaseProvider(ABC):
    """Interface every AI provider implements."""

    name: str = "base"

    @abstractmethod
    async def generate(
        self,
        model: str,
        contents: list[ContentPart],
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        """Non-streaming generation. Tool calls requested by the model are
        returned in ``GenerationResult.tool_calls`` (the executor feeds their
        results back in a follow-up round)."""

    @abstractmethod
    def generate_stream(
        self,
        model: str,
        contents: list[ContentPart],
        config: GenerationConfig | None = None,
    ) -> AsyncIterator[GenerationEvent]:
        """Streaming generation yielding ``GenerationEvent`` items."""

    @abstractmethod
    def build_tools(self, tool_configs: list[dict[str, Any]]) -> list[Any]:
        """Convert internal tool definitions (from the tool registry) into
        provider-native tool objects."""

    @abstractmethod
    async def parts_for_attachment(
        self,
        attachment: Any,
        data: bytes,
    ) -> list[dict[str, Any]]:
        """Convert an uploaded media attachment into provider parts.

        ``attachment`` is an ``AttachmentMetadata`` instance; implementations
        may mutate it (e.g. caching a provider file URI) — callers persist it.
        """

    @abstractmethod
    async def transcribe_audio(
        self,
        model: str,
        audio_bytes: bytes,
        mime_type: str,
        prompt: str,
    ) -> tuple[str, int]:
        """Transcribe audio bytes, returning ``(transcript, tokens_used)``."""

    def supports(self, capability: str) -> bool:
        """Check if this provider natively supports a given capability."""
        return True

    def supports_model(self, model: str) -> bool:
        """Check if this provider can execute requests for the given model ID."""
        return True

    def normalize_error(self, exc: Exception) -> Exception:
        """Translate provider-specific SDK exceptions into normalized ProviderError subclasses."""
        if isinstance(exc, ProviderError):
            return exc
        return ProviderGenerationError(str(exc) or SAFE_GENERATION_ERROR)

    def normalize_usage(self, usage: Any) -> int:
        """Extract or normalize total tokens count from provider-specific usage object."""
        if isinstance(usage, int):
            return usage
        if hasattr(usage, "total_token_count"):
            return usage.total_token_count or 0
        if hasattr(usage, "total_tokens"):
            return usage.total_tokens or 0
        if isinstance(usage, dict):
            return usage.get("total_tokens") or usage.get("total_token_count") or 0
        return 0

    @staticmethod
    def is_retryable_error(exc: Exception) -> bool:
        """Determine whether a generation failure is safe to retry on a fallback model."""
        if isinstance(
            exc,
            (
                ProviderRateLimitError,
                ProviderOverloadedError,
                ProviderTimeoutError,
                ProviderConnectionError,
                ProviderModelNotFoundError,
            ),
        ):
            return True
        if isinstance(exc, (ProviderAuthError, ProviderInvalidRequestError, ProviderUnsupportedError)):
            return False
        if isinstance(exc, ProviderGenerationError):
            return exc.status_code in (404, 429, 500, 502, 503, 504)
        return False

    async def delete_file(self, file_uri: str) -> None:
        """Delete an uploaded remote file if the provider supports file storage."""


def is_retryable_provider_error(exc: Exception) -> bool:
    """Convenience helper to check if a provider exception is retryable."""
    return BaseProvider.is_retryable_error(exc)
