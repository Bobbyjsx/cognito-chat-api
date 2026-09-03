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


class ProviderError(Exception):
    """Base class for all provider failures surfaced to callers."""


class ProviderModelNotFoundError(ProviderError):
    """The requested model does not exist on the provider."""


class ProviderGenerationError(ProviderError):
    """Generation failed with a provider-side error."""

    def __init__(self, message: str = SAFE_GENERATION_ERROR, status_code: int = 500):
        super().__init__(message)
        self.status_code = status_code


def classify_provider_error(exc: Exception) -> tuple[int, str, str]:
    """Return ``(status_code, error_code, message)`` for a raised exception.

    Mirrors the behaviour of the historical ``extract_genai_error`` helper so
    HTTP semantics stay stable, while keeping SDK-specific exceptions inside
    the provider layer.
    """
    if isinstance(exc, ProviderModelNotFoundError):
        return 404, ERROR_CODE_MODEL_NOT_FOUND, str(exc) or SAFE_GENERATION_ERROR
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

    async def delete_file(self, file_uri: str) -> None:
        """Delete an uploaded file from the provider (optional)."""
