"""Anthropic Claude provider via AWS Bedrock.

Directly communicates with AWS Bedrock via the official ``anthropic`` SDK's
``AsyncAnthropicBedrock`` client. Normalizes Anthropic-specific formatting,
streaming events, tool invocations, and error structures into Cognito's
provider-agnostic abstractions.

Note:
Audio transcription models are selected by the backend configuration
(`stt_model` in AppConfigDB, e.g. `gemini-3.1-flash-lite`). Claude models
do not serve audio transcription.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import uuid
from collections.abc import AsyncIterator
from typing import Any

from anthropic import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)
from anthropic.lib.bedrock import AsyncAnthropicBedrock

from app.models.attachments import AttachmentMetadata
from app.providers.base import (
    TOOL_KIND_FUNCTION,
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
)

logger = logging.getLogger(__name__)

# Supported MIME types for Claude vision and document input
SUPPORTED_IMAGE_MIMES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
SUPPORTED_DOCUMENT_MIMES = {"application/pdf"}

_ANTHROPIC_ERROR_HANDLED = (APIError, APIConnectionError, APITimeoutError)


class ClaudeProvider(BaseProvider):
    """Concrete provider for Anthropic Claude models via AWS Bedrock."""

    name = "anthropic"

    # Friendly model IDs mapped to AWS Bedrock inference profile identifiers
    MODEL_MAP: dict[str, str] = {
        "claude-sonnet-4-5": "eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "claude-sonnet-4.5": "eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "claude-sonnet-4-5@20250929": "eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "claude-haiku-4-5": "eu.anthropic.claude-haiku-4-5-20251001-v1:0",
        "claude-haiku-4.5": "eu.anthropic.claude-haiku-4-5-20251001-v1:0",
        "claude-3-5-haiku": "eu.anthropic.claude-haiku-4-5-20251001-v1:0",
        "claude-3.5-haiku": "eu.anthropic.claude-haiku-4-5-20251001-v1:0",
        "claude-3-5-haiku-20241022": "eu.anthropic.claude-haiku-4-5-20251001-v1:0",
        "claude-3-5-sonnet": "eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "claude-3.5-sonnet": "eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "claude-3-7-sonnet": "eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "claude-3.7-sonnet": "eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "claude-3-opus": "eu.anthropic.claude-opus-4-5-20251101-v1:0",
        "claude-3-haiku": "eu.anthropic.claude-3-haiku-20240307-v1:0",
    }

    def __init__(
        self,
        api_key: str | None = None,
        aws_region: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        client: AsyncAnthropicBedrock | None = None,
        default_max_tokens: int = 8192,
        **kwargs: Any,
    ):
        self._client = client
        self._lazy_client: AsyncAnthropicBedrock | None = None
        self.aws_region = aws_region or kwargs.get("region") or os.getenv("AWS_REGION") or "eu-west-1"
        self.api_key = (
            api_key
            or kwargs.get("aws_bedrock_api_key")
            or os.getenv("AWS_BEARER_TOKEN_BEDROCK")
            or os.getenv("AWS_BEDROCK_API_KEY")
            or os.getenv("AWS_API_KEY")
        )
        self.aws_access_key_id = aws_access_key_id or kwargs.get("aws_access_key_id") or os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = (
            aws_secret_access_key or kwargs.get("aws_secret_access_key") or os.getenv("AWS_SECRET_ACCESS_KEY")
        )
        self.aws_session_token = aws_session_token or kwargs.get("aws_session_token") or os.getenv("AWS_SESSION_TOKEN")
        self.default_max_tokens = default_max_tokens

    # ── client lifecycle ─────────────────────────────────────────────────────

    @property
    def client(self) -> AsyncAnthropicBedrock:
        if self._client is not None:
            return self._client
        if self._lazy_client is None:
            self._lazy_client = self._init_client()
        return self._lazy_client

    def _init_client(self) -> AsyncAnthropicBedrock:
        region = self.aws_region or "eu-west-1"
        token = self.api_key

        if token and not os.getenv("AWS_BEARER_TOKEN_BEDROCK"):
            os.environ["AWS_BEARER_TOKEN_BEDROCK"] = token

        kwargs: dict[str, Any] = {
            "aws_region": region,
        }
        if token:
            kwargs["api_key"] = token
        if self.aws_access_key_id:
            kwargs["aws_access_key"] = self.aws_access_key_id
        if self.aws_secret_access_key:
            kwargs["aws_secret_key"] = self.aws_secret_access_key
        if self.aws_session_token:
            kwargs["aws_session_token"] = self.aws_session_token

        logger.info("Initialized ClaudeProvider with AWS Bedrock client (region=%s)", region)
        return AsyncAnthropicBedrock(**kwargs)

    def resolve_model_id(self, model: str) -> str:
        """Resolve friendly model identifier into Bedrock inference profile ID."""
        cleaned = model.strip()
        if cleaned.startswith(("eu.anthropic", "us.anthropic", "global.anthropic", "anthropic.")):
            return cleaned
        mapped = self.MODEL_MAP.get(cleaned.lower(), cleaned)
        reg = (self.aws_region or "").lower()
        if reg.startswith("us-") and mapped.startswith("eu.anthropic"):
            mapped = mapped.replace("eu.anthropic", "us.anthropic", 1)
        return mapped

    # ── conversion helpers ───────────────────────────────────────────────────

    def _to_sdk_messages_and_system(
        self,
        contents: list[ContentPart],
        system_instruction: str | None = None,
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Convert provider-agnostic ContentParts into Anthropic (system, messages) tuple.

        Enforces Anthropic message formatting rules:
        - Roles must be 'user' or 'assistant'.
        - Message turns must alternate roles (consecutive turns of same role are merged).
        - First message must have role 'user'.
        """
        raw_messages: list[dict[str, Any]] = []

        for content in contents:
            # Map Cognito role "model" -> Anthropic "assistant"
            role = "assistant" if content.role in ("model", "assistant") else "user"
            blocks: list[dict[str, Any]] = []

            for part in content.parts:
                if "text" in part:
                    text_val = part["text"]
                    if text_val:  # Anthropic rejects empty text blocks in some contexts
                        blocks.append({"type": "text", "text": text_val})

                elif "inline_data" in part:
                    inline = part["inline_data"]
                    mime = inline.get("mime_type", "")
                    data = inline.get("data", "")
                    if mime in SUPPORTED_IMAGE_MIMES:
                        blocks.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": mime,
                                    "data": data,
                                },
                            }
                        )
                    elif mime in SUPPORTED_DOCUMENT_MIMES:
                        blocks.append(
                            {
                                "type": "document",
                                "source": {
                                    "type": "base64",
                                    "media_type": "application/pdf",
                                    "data": data,
                                },
                            }
                        )
                    else:
                        # Fallback for textual / other data
                        try:
                            decoded = base64.b64decode(data).decode("utf-8", errors="replace")
                            blocks.append({"type": "text", "text": decoded})
                        except Exception:
                            blocks.append({"type": "text", "text": f"[Attachment: {mime}]"})

                elif "file_data" in part:
                    file_data = part["file_data"]
                    blocks.append({"type": "text", "text": f"[File: {file_data.get('file_uri')}]"})

                elif "function_call" in part:
                    fc = part["function_call"]
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": fc.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                            "name": fc["name"],
                            "input": fc.get("args", {}),
                        }
                    )

                elif "function_response" in part:
                    fr = part["function_response"]
                    resp_val = fr.get("response", {})
                    content_str = json.dumps(resp_val) if isinstance(resp_val, (dict, list)) else str(resp_val)
                    blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": fr.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                            "content": content_str,
                        }
                    )

            if blocks:
                raw_messages.append({"role": role, "content": blocks})

        if not raw_messages:
            raw_messages.append({"role": "user", "content": [{"type": "text", "text": "Hello"}]})

        # Merge consecutive messages of the same role to adhere to strict alternation
        merged_messages: list[dict[str, Any]] = []
        for msg in raw_messages:
            if not merged_messages:
                # Ensure the first message is from 'user'
                if msg["role"] != "user":
                    merged_messages.append(
                        {"role": "user", "content": [{"type": "text", "text": "Begin conversation."}]}
                    )
                merged_messages.append(msg)
            elif merged_messages[-1]["role"] == msg["role"]:
                merged_messages[-1]["content"].extend(msg["content"])
            else:
                merged_messages.append(msg)

        return system_instruction, merged_messages

    def _to_sdk_params(
        self,
        model: str,
        contents: list[ContentPart],
        config: GenerationConfig | None = None,
    ) -> dict[str, Any]:
        """Prepare keyword arguments for anthropic messages.create or messages.stream."""
        if config is None:
            config = GenerationConfig()

        system_instruction, messages = self._to_sdk_messages_and_system(
            contents=contents,
            system_instruction=config.system_instruction,
        )

        native_model = self.resolve_model_id(model)

        params: dict[str, Any] = {
            "model": native_model,
            "messages": messages,
            "max_tokens": self.default_max_tokens,
        }

        if system_instruction:
            params["system"] = system_instruction

        # Configure tools
        if config.tool_configs:
            tools = self.build_tools(config.tool_configs)
            if tools:
                params["tools"] = tools

        # Configure reasoning / extended thinking
        # For Claude models with thinking support (e.g. Claude 3.7 Sonnet, Claude Sonnet 4.5):
        # thinking parameter requires {"type": "enabled", "budget_tokens": N}
        # and temperature must be 1.0 (or omitted/default).
        supports_thinking = (
            "3-7" in native_model
            or "3.7" in native_model
            or "sonnet-4" in native_model
            or "4-5" in native_model
            or "4.5" in native_model
            or "opus-4" in native_model
        )
        if supports_thinking and config.thinking_budget is not None and config.thinking_budget > 0:
            budget = max(1024, config.thinking_budget)
            params["thinking"] = {"type": "enabled", "budget_tokens": budget}
            # max_tokens must be greater than budget_tokens
            params["max_tokens"] = max(budget + 4096, self.default_max_tokens)
        return params

    # ── error handling ───────────────────────────────────────────────────────

    def normalize_error(self, exc: Exception) -> Exception:
        """Map Anthropic Bedrock SDK exceptions into Cognito normalized ProviderError subclasses."""
        if isinstance(exc, ProviderError):
            return exc

        msg = str(exc) or "Claude Bedrock model generation failed."
        if isinstance(exc, NotFoundError):
            return ProviderModelNotFoundError(msg or "Model not found on AWS Bedrock.")
        if isinstance(exc, RateLimitError):
            return ProviderRateLimitError(msg or "AWS Bedrock rate limit exceeded.", status_code=429)
        if isinstance(exc, (AuthenticationError, PermissionDeniedError)):
            return ProviderAuthError(
                f"AWS Bedrock authentication failed: {msg}. Please verify AWS_BEARER_TOKEN_BEDROCK or IAM credentials.",
                status_code=403 if isinstance(exc, PermissionDeniedError) else 401,
            )
        if isinstance(exc, BadRequestError):
            if "context_length_exceeded" in msg.lower() or "prompt is too long" in msg.lower():
                return ProviderInvalidRequestError(msg, status_code=400)
            return ProviderInvalidRequestError(msg, status_code=400)
        if isinstance(exc, InternalServerError):
            return ProviderOverloadedError(msg or "AWS Bedrock service overloaded.", status_code=503)
        if isinstance(exc, APITimeoutError):
            return ProviderTimeoutError(msg or "AWS Bedrock request timed out.", status_code=504)
        if isinstance(exc, APIConnectionError):
            return ProviderConnectionError(msg or "Failed to connect to AWS Bedrock endpoint.", status_code=502)
        if isinstance(exc, APIError):
            status = getattr(exc, "status_code", 500) or 500
            return ProviderGenerationError(msg, status_code=int(status))

        return ProviderGenerationError(msg, status_code=500)

    # ── capabilities ─────────────────────────────────────────────────────────

    def supports(self, capability: str) -> bool:
        """Check capability support.

        Note: Audio transcription models are selected by backend configuration
        (`stt_model` in AppConfigDB, default: `gemini-3.1-flash-lite`). Claude
        models on Bedrock do not handle audio transcription.
        """
        if capability in ("vision", "tools", "reasoning", "structured_output"):
            return True
        return capability not in ("audio", "audio_transcription", "google_search", "code_execution")

    def supports_model(self, model: str) -> bool:
        lower = model.lower()
        return lower.startswith("claude") or "anthropic" in lower or lower in self.MODEL_MAP

    # ── tool construction ────────────────────────────────────────────────────

    def build_tools(self, tool_configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert internal tool definitions into Anthropic tool schema objects."""
        tools: list[dict[str, Any]] = []
        for cfg in tool_configs:
            kind = cfg.get("kind", TOOL_KIND_FUNCTION)
            if kind == TOOL_KIND_FUNCTION:
                tools.append(
                    {
                        "name": cfg["name"],
                        "description": cfg.get("description", ""),
                        "input_schema": cfg.get("schema") or {"type": "object", "properties": {}},
                    }
                )
            # Server-specific tools (code_execution / google_search) are Gemini-specific
            # Claude gracefully ignores them or maps function definitions
        return tools

    # ── non-streaming generation ─────────────────────────────────────────────

    async def generate(
        self,
        model: str,
        contents: list[ContentPart],
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        params = self._to_sdk_params(model=model, contents=contents, config=config)
        try:
            if params.get("thinking") or params.get("max_tokens", 0) > 21333:
                async with self.client.messages.stream(**params) as stream:
                    response = await stream.get_final_message()
            else:
                response = await self.client.messages.create(**params)
        except _ANTHROPIC_ERROR_HANDLED as exc:
            raise self.normalize_error(exc) from exc
        except Exception as exc:
            raise self.normalize_error(exc) from exc

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in response.content:
            if getattr(block, "type", "") == "text":
                text_parts.append(getattr(block, "text", "") or "")
            elif getattr(block, "type", "") == "tool_use":
                args = getattr(block, "input", {})
                tool_calls.append(
                    ToolCall(
                        id=getattr(block, "id", f"call_{uuid.uuid4().hex[:8]}"),
                        name=getattr(block, "name", ""),
                        args=args if isinstance(args, dict) else {},
                        kind=TOOL_KIND_FUNCTION,
                    )
                )

        total_tokens = 0
        if getattr(response, "usage", None):
            usage = response.usage
            input_tokens = getattr(usage, "input_tokens", 0) or 0
            output_tokens = getattr(usage, "output_tokens", 0) or 0
            total_tokens = input_tokens + output_tokens

        return GenerationResult(
            text="".join(text_parts),
            total_tokens=total_tokens,
            tool_calls=tool_calls,
        )

    # ── streaming generation ─────────────────────────────────────────────────

    async def generate_stream(
        self,
        model: str,
        contents: list[ContentPart],
        config: GenerationConfig | None = None,
    ) -> AsyncIterator[GenerationEvent]:
        params = self._to_sdk_params(model=model, contents=contents, config=config)

        try:
            async with self.client.messages.stream(**params) as stream:
                current_tool_id: str | None = None
                current_tool_name: str | None = None
                tool_json_chunks: list[str] = []

                async for event in stream:
                    event_type = getattr(event, "type", "")

                    # 1. Delta events (text, thinking, tool json)
                    if event_type == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        if delta is not None:
                            delta_type = getattr(delta, "type", "")
                            if delta_type == "text_delta":
                                text = getattr(delta, "text", "")
                                if text:
                                    yield GenerationEvent(type="text", token=text)
                            elif delta_type == "thinking_delta":
                                thought = getattr(delta, "thinking", "")
                                if thought:
                                    yield GenerationEvent(type="reasoning", token=thought)
                            elif delta_type == "input_json_delta":
                                partial_json = getattr(delta, "partial_json", "")
                                if partial_json:
                                    tool_json_chunks.append(partial_json)

                    # 2. Tool use start
                    elif event_type == "content_block_start":
                        content_block = getattr(event, "content_block", None)
                        if content_block and getattr(content_block, "type", "") == "tool_use":
                            current_tool_id = getattr(content_block, "id", f"call_{uuid.uuid4().hex[:8]}")
                            current_tool_name = getattr(content_block, "name", "")
                            tool_json_chunks = []

                    # 3. Tool use stop -> emit ToolCall event
                    elif event_type == "content_block_stop":
                        if current_tool_id and current_tool_name:
                            full_json_str = "".join(tool_json_chunks)
                            try:
                                parsed_args = json.loads(full_json_str) if full_json_str.strip() else {}
                            except Exception:
                                logger.warning("Failed to parse tool JSON args: %s", full_json_str)
                                parsed_args = {}

                            yield GenerationEvent(
                                type="tool_call",
                                tool_call=ToolCall(
                                    id=current_tool_id,
                                    name=current_tool_name,
                                    args=parsed_args,
                                    kind=TOOL_KIND_FUNCTION,
                                ),
                            )
                            current_tool_id = None
                            current_tool_name = None
                            tool_json_chunks = []

                # Extract final message usage
                final_msg = await stream.get_final_message()
                if final_msg and getattr(final_msg, "usage", None):
                    usage = final_msg.usage
                    input_tokens = getattr(usage, "input_tokens", 0) or 0
                    output_tokens = getattr(usage, "output_tokens", 0) or 0
                    total_tokens = input_tokens + output_tokens
                    yield GenerationEvent(type="usage", total_tokens=total_tokens)

        except _ANTHROPIC_ERROR_HANDLED as exc:
            raise self.normalize_error(exc) from exc
        except Exception as exc:
            raise self.normalize_error(exc) from exc

    # ── attachments ──────────────────────────────────────────────────────────

    async def parts_for_attachment(
        self,
        attachment: AttachmentMetadata,
        data: bytes,
    ) -> list[dict[str, Any]]:
        mime_type = attachment.mime_type
        b64_str = base64.standard_b64encode(data).decode("utf-8")

        if attachment.type == "image" and mime_type in SUPPORTED_IMAGE_MIMES:
            return [{"inline_data": {"mime_type": mime_type, "data": b64_str}}]

        if attachment.type == "pdf" or mime_type == "application/pdf":
            return [{"inline_data": {"mime_type": "application/pdf", "data": b64_str}}]

        # Audio / Video not supported directly by Claude API
        logger.warning(
            "Attachment '%s' (%s) is not directly supported by Claude vision/document API; sending text description fallback.",
            attachment.filename,
            attachment.mime_type,
        )
        return [{"text": f"[Attached file: {attachment.filename} ({attachment.mime_type}, {len(data)} bytes)]"}]

    # ── transcription ────────────────────────────────────────────────────────

    async def transcribe_audio(
        self,
        model: str,
        audio_bytes: bytes,
        mime_type: str,
        prompt: str,
    ) -> tuple[str, int]:
        raise ProviderUnsupportedError("Claude provider does not support direct audio transcription. Use Gemini STT.")
