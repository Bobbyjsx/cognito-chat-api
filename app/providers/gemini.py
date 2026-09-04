"""Google Gemini provider.

Every direct interaction with the ``google.genai`` SDK lives in this module.
The rest of the application works with the provider-agnostic types from
``app.providers.base``.
"""

from __future__ import annotations

import base64
import logging
import tempfile
import uuid
from collections import deque
from collections.abc import AsyncIterator
from typing import Any

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from app.models.attachments import AttachmentMetadata
from app.providers.base import (
    TOOL_KIND_FUNCTION,
    TOOL_KIND_SERVER,
    BaseProvider,
    ContentPart,
    GenerationConfig,
    GenerationEvent,
    GenerationResult,
    ProviderError,
    ProviderGenerationError,
    ProviderModelNotFoundError,
    ToolCall,
    ToolResult,
)

logger = logging.getLogger(__name__)

INLINE_DATA_LIMIT = 20 * 1024 * 1024  # Gemini API inline data cap
AUDIO_INLINE_LIMIT = 9 * 1024 * 1024  # conservative cap for inline audio
VIDEO_TYPES = {"video"}

_GEMINI_ERROR_HANDLED = (genai_errors.APIError,)


class GeminiProvider(BaseProvider):
    """Concrete provider for Google's Gemini models via the google-genai SDK."""

    name = "gemini"

    def __init__(self, api_key: str | None = None, client: genai.Client | None = None):
        self._api_key = api_key
        self._client = client
        self._lazy_client = None  # created on first use so tests can inject fakes

    # ── client lifecycle ─────────────────────────────────────────────────────

    @property
    def client(self) -> genai.Client:
        if self._client is not None:
            return self._client
        if self._lazy_client is None:
            self._lazy_client = genai.Client(api_key=self._api_key) if self._api_key else genai.Client()
        return self._lazy_client

    # ── conversion helpers ───────────────────────────────────────────────────

    def _to_sdk_contents(self, contents: list[ContentPart]) -> list[types.Content]:
        sdk_contents: list[types.Content] = []
        for content in contents:
            parts: list[types.Part] = []
            for part in content.parts:
                if "text" in part:
                    parts.append(types.Part.from_text(text=part["text"]))
                elif "inline_data" in part:
                    inline = part["inline_data"]
                    parts.append(
                        types.Part(
                            inline_data=types.Blob(
                                mime_type=inline["mime_type"],
                                data=inline["data"],
                            )
                        )
                    )
                elif "file_data" in part:
                    file_data = part["file_data"]
                    parts.append(
                        types.Part(
                            file_data=types.FileData(
                                file_uri=file_data["file_uri"],
                                mime_type=file_data.get("mime_type"),
                            )
                        )
                    )
                elif "function_call" in part:
                    fc = part["function_call"]
                    thought_sig = part.get("thought_signature") or (
                        fc.get("thought_signature") if isinstance(fc, dict) else None
                    )
                    part_kwargs: dict[str, Any] = {
                        "function_call": types.FunctionCall(
                            id=fc["id"],
                            name=fc["name"],
                            args=fc.get("args", {}),
                        )
                    }
                    if thought_sig is not None:
                        part_kwargs["thought_signature"] = thought_sig
                    parts.append(types.Part(**part_kwargs))
                elif "function_response" in part:
                    fr = part["function_response"]
                    parts.append(
                        types.Part(
                            function_response=types.FunctionResponse(
                                id=fr["id"],
                                name=fr["name"],
                                response=fr.get("response", {}),
                            )
                        )
                    )
            sdk_contents.append(types.Content(role=content.role, parts=parts))
        return sdk_contents

    def _to_sdk_config(self, config: GenerationConfig | None) -> types.GenerateContentConfig:
        if config is None:
            config = GenerationConfig()

        thinking_config = None
        if config.thinking_budget is not None and config.thinking_budget > 0:
            thinking_config = types.ThinkingConfig(
                thinking_budget=config.thinking_budget,
                include_thoughts=config.include_thoughts,
            )

        tools = self.build_tools(config.tool_configs) if config.tool_configs else None

        # Disable SDK-level Automatic Function Calling (AFC) because ToolExecutor
        # explicitly handles tool calls, SSE event streaming, and conversational tool execution.
        afc_config = types.AutomaticFunctionCallingConfig(disable=True)

        return types.GenerateContentConfig(
            system_instruction=config.system_instruction,
            thinking_config=thinking_config,
            tools=tools,
            automatic_function_calling=afc_config,
        )

    def _extract_function_calls(self, content: Any) -> list[ToolCall]:
        calls: list[ToolCall] = []
        for part in getattr(content, "parts", []) or []:
            function_call = getattr(part, "function_call", None)
            if function_call is None:
                continue
            calls.append(
                ToolCall(
                    id=function_call.id or f"call_{uuid.uuid4().hex[:8]}",
                    name=function_call.name,
                    args=function_call.args or {},
                    kind=TOOL_KIND_FUNCTION,
                    thought_signature=getattr(part, "thought_signature", None),
                )
            )
        return calls

    def normalize_error(self, exc: Exception) -> Exception:
        if isinstance(exc, genai_errors.APIError):
            status = getattr(exc, "code", None) or 500
            message = getattr(exc, "message", None) or str(exc) or ""
            if status == 404 or getattr(exc, "status", None) == "NOT_FOUND":
                return ProviderModelNotFoundError(message or "Model not found.")
            if status == 429 or getattr(exc, "status", None) == "RESOURCE_EXHAUSTED":
                from app.providers.base import ProviderRateLimitError

                return ProviderRateLimitError(message or "Gemini quota / rate limit exceeded.", status_code=429)
            if status in (401, 403) or getattr(exc, "status", None) == "PERMISSION_DENIED":
                from app.providers.base import ProviderAuthError

                return ProviderAuthError(message or "Gemini authentication failed.", status_code=int(status))
            if status == 400 or getattr(exc, "status", None) == "INVALID_ARGUMENT":
                from app.providers.base import ProviderInvalidRequestError

                return ProviderInvalidRequestError(message or "Invalid Gemini request.", status_code=400)
            if status == 503 or getattr(exc, "status", None) == "UNAVAILABLE":
                from app.providers.base import ProviderOverloadedError

                return ProviderOverloadedError(message or "Gemini service is temporarily unavailable.", status_code=503)
            if status == 504 or getattr(exc, "status", None) == "DEADLINE_EXCEEDED":
                from app.providers.base import ProviderTimeoutError

                return ProviderTimeoutError(message or "Gemini request timed out.", status_code=504)
            return ProviderGenerationError(message or "Model generation failed.", status_code=int(status))
        if isinstance(exc, ProviderError):
            return exc
        return ProviderGenerationError(str(exc) or "Model generation failed.", status_code=500)

    def _wrap_error(self, exc: Exception) -> Exception:
        return self.normalize_error(exc)

    def supports(self, capability: str) -> bool:
        if capability in (
            "audio",
            "audio_transcription",
            "vision",
            "tools",
            "reasoning",
            "web_search",
            "code_execution",
        ):
            return True
        return True

    def supports_model(self, model: str) -> bool:
        return model.startswith("gemini") or "imagen" in model or "veo" in model

    # ── tool construction ────────────────────────────────────────────────────

    def build_tools(self, tool_configs: list[dict[str, Any]]) -> list[Any]:
        tools: list[types.Tool] = []
        for cfg in tool_configs:
            kind = cfg.get("kind")
            if kind == "code_execution":
                tools.append(types.Tool(code_execution=types.ToolCodeExecution()))
            elif kind == "google_search":
                tools.append(self._build_google_search_tool())
            elif kind == TOOL_KIND_FUNCTION:
                tools.append(
                    types.Tool(
                        function_declarations=[
                            types.FunctionDeclaration(
                                name=cfg["name"],
                                description=cfg.get("description", ""),
                                parameters=types.Schema(**cfg["schema"]) if cfg.get("schema") else None,
                            )
                        ]
                    )
                )
        return tools

    def _build_google_search_tool(self) -> types.Tool:
        """Construct the Google Search grounding tool.

        Newer SDK versions expose ``types.GoogleSearch``; older ones use the
        deprecated ``types.ToolGoogleSearch`` alias. Try both.
        """
        for klass in ("GoogleSearch", "ToolGoogleSearch"):
            cls = getattr(types, klass, None)
            if cls is not None:
                try:
                    return types.Tool(google_search=cls())
                except TypeError:
                    continue
        return types.Tool(google_search=types.GoogleSearch())

    # ── generation ───────────────────────────────────────────────────────────

    async def generate(
        self,
        model: str,
        contents: list[ContentPart],
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        try:
            response = await self.client.aio.models.generate_content(
                model=model,
                contents=self._to_sdk_contents(contents),
                config=self._to_sdk_config(config),
            )
        except _GEMINI_ERROR_HANDLED as exc:
            raise self._wrap_error(exc) from exc

        text = getattr(response, "text", None) or ""
        total_tokens = 0
        if getattr(response, "usage_metadata", None):
            total_tokens = getattr(response.usage_metadata, "total_token_count", 0) or 0

        tool_calls: list[ToolCall] = []
        if response.candidates:
            cand = response.candidates[0]
            content = getattr(cand, "content", None)
            if content is not None:
                tool_calls = self._extract_function_calls(content)
            grounding = getattr(cand, "grounding_metadata", None)
            if grounding:
                queries = getattr(grounding, "web_search_queries", None) or []
                query_text = queries[0] if queries else "Web search"
                tool_calls.append(
                    ToolCall(
                        id=f"call_{uuid.uuid4().hex[:8]}",
                        name="google_search",
                        args={"query": query_text},
                        kind=TOOL_KIND_SERVER,
                    )
                )

        return GenerationResult(text=text, total_tokens=total_tokens, tool_calls=tool_calls)

    async def generate_stream(
        self,
        model: str,
        contents: list[ContentPart],
        config: GenerationConfig | None = None,
    ) -> AsyncIterator[GenerationEvent]:
        try:
            stream = await self.client.aio.models.generate_content_stream(
                model=model,
                contents=self._to_sdk_contents(contents),
                config=self._to_sdk_config(config),
            )
        except _GEMINI_ERROR_HANDLED as exc:
            raise self._wrap_error(exc) from exc

        search_announced = False
        search_call_id = f"call_{uuid.uuid4().hex[:8]}"
        try:
            async for chunk in stream:
                for event in self._extract_stream_events(chunk, search_announced, search_call_id):
                    if event.type == "tool_call" and event.tool_call and event.tool_call.name == "google_search":
                        search_announced = True
                    yield event
        except _GEMINI_ERROR_HANDLED as exc:
            raise self._wrap_error(exc) from exc

    def _extract_stream_events(
        self,
        chunk: Any,
        search_announced: bool,
        search_call_id: str | None = None,
    ) -> list[GenerationEvent]:
        events: list[GenerationEvent] = []
        code_tool_ids: deque[str] = deque()

        candidates = getattr(chunk, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if not content:
                continue

            for part in content.parts or []:
                text = getattr(part, "text", None)
                thought = getattr(part, "thought", None)
                executable_code = getattr(part, "executable_code", None)
                code_execution_result = getattr(part, "code_execution_result", None)
                function_call = getattr(part, "function_call", None)

                if thought:
                    thought_str = text if text else (thought if isinstance(thought, str) else "")
                    if thought_str:
                        events.append(GenerationEvent(type="reasoning", token=thought_str))
                elif text:
                    events.append(GenerationEvent(type="text", token=text))

                if function_call is not None:
                    events.append(
                        GenerationEvent(
                            type="tool_call",
                            tool_call=ToolCall(
                                id=function_call.id or f"call_{uuid.uuid4().hex[:8]}",
                                name=function_call.name,
                                args=function_call.args or {},
                                kind=TOOL_KIND_FUNCTION,
                                thought_signature=getattr(part, "thought_signature", None),
                            ),
                        )
                    )

                if executable_code is not None:
                    call_id = f"call_{uuid.uuid4().hex[:8]}"
                    code_tool_ids.append(call_id)
                    language = getattr(executable_code, "language", None)
                    events.append(
                        GenerationEvent(
                            type="tool_call",
                            tool_call=ToolCall(
                                id=call_id,
                                name="code_execution",
                                args={
                                    "language": str(language).lower() if language else "python",
                                    "code": getattr(executable_code, "code", ""),
                                },
                                kind=TOOL_KIND_SERVER,
                            ),
                        )
                    )

                if code_execution_result is not None:
                    call_id = code_tool_ids.popleft() if code_tool_ids else f"call_{uuid.uuid4().hex[:8]}"
                    outcome = getattr(code_execution_result, "outcome", None)
                    events.append(
                        GenerationEvent(
                            type="tool_result",
                            tool_result=ToolResult(
                                id=call_id,
                                name="code_execution",
                                output={
                                    "outcome": str(outcome) if outcome else "OUTCOME_OK",
                                    "output": getattr(code_execution_result, "output", ""),
                                },
                                kind=TOOL_KIND_SERVER,
                            ),
                        )
                    )

        if getattr(chunk, "usage_metadata", None):
            total_tokens = getattr(chunk.usage_metadata, "total_token_count", 0) or 0
            events.append(GenerationEvent(type="usage", total_tokens=total_tokens))

        events.extend(self._extract_grounding_events(chunk, search_announced, search_call_id))
        return events

    def _extract_grounding_events(
        self,
        chunk: Any,
        search_announced: bool,
        search_call_id: str | None = None,
    ) -> list[GenerationEvent]:
        """Surface Google Search grounding as tool_call/tool_result events."""
        grounding = getattr(chunk, "grounding_metadata", None)
        if grounding is None:
            for cand in getattr(chunk, "candidates", None) or []:
                cand_grounding = getattr(cand, "grounding_metadata", None)
                if cand_grounding is not None:
                    grounding = cand_grounding
                    break

        if grounding is None:
            return []

        sources = []
        for gchunk in getattr(grounding, "grounding_chunks", None) or []:
            web = getattr(gchunk, "web", None)
            if web is not None:
                sources.append(
                    {
                        "title": getattr(web, "title", None) or "Web Source",
                        "uri": getattr(web, "uri", None) or "",
                    }
                )
        if not sources:
            return []

        queries = getattr(grounding, "web_search_queries", None) or []
        query_text = queries[0] if queries else "Web search"
        call_id = search_call_id or f"call_{uuid.uuid4().hex[:8]}"

        events: list[GenerationEvent] = []
        if not search_announced:
            events.append(
                GenerationEvent(
                    type="tool_call",
                    tool_call=ToolCall(
                        id=call_id,
                        name="google_search",
                        args={"query": query_text},
                        kind=TOOL_KIND_SERVER,
                    ),
                )
            )
        events.append(
            GenerationEvent(
                type="tool_result",
                tool_result=ToolResult(
                    id=call_id,
                    name="google_search",
                    output={"sources": sources, "queries": queries},
                    kind=TOOL_KIND_SERVER,
                ),
            )
        )
        return events

    # ── attachments ──────────────────────────────────────────────────────────

    async def parts_for_attachment(
        self,
        attachment: AttachmentMetadata,
        data: bytes,
    ) -> list[dict[str, Any]]:
        mime_type = attachment.mime_type
        size = len(data)

        if attachment.type == "image" or attachment.type == "pdf":
            if size <= INLINE_DATA_LIMIT:
                return [self._inline_part(data, mime_type)]
            file_uri = await self._ensure_file_uri(attachment, data, mime_type)
            return [self._file_data_part(file_uri, mime_type)]

        if attachment.type == "audio":
            if size <= AUDIO_INLINE_LIMIT:
                return [self._inline_part(data, mime_type)]
            file_uri = await self._ensure_file_uri(attachment, data, mime_type)
            return [self._file_data_part(file_uri, mime_type)]

        if attachment.type in VIDEO_TYPES:
            file_uri = await self._ensure_file_uri(attachment, data, mime_type)
            return [self._file_data_part(file_uri, mime_type)]

        logger.warning("Attachment type %s not handled by provider; sending as text fallback", attachment.type)
        return [{"text": ""}]

    def _inline_part(self, data: bytes, mime_type: str) -> dict[str, Any]:
        return {
            "inline_data": {
                "mime_type": mime_type,
                "data": base64.standard_b64encode(data).decode("utf-8"),
            }
        }

    def _file_data_part(self, file_uri: str, mime_type: str) -> dict[str, Any]:
        return {"file_data": {"file_uri": file_uri, "mime_type": mime_type}}

    async def _ensure_file_uri(self, attachment: AttachmentMetadata, data: bytes, mime_type: str) -> str:
        """Upload a media file to the Gemini Files API once, caching the URI.

        The URI is stored back on the attachment metadata so subsequent turns
        reuse it instead of re-uploading.
        """
        if attachment.gemini_file_uri:
            return attachment.gemini_file_uri

        suffix = attachment.filename.rsplit(".", 1)[-1] if "." in attachment.filename else ""
        with tempfile.NamedTemporaryFile(suffix=f".{suffix}", delete=True) as tmp:
            tmp.write(data)
            tmp.flush()
            uploaded = await self.client.aio.files.upload(
                file=tmp.name,
                config=types.UploadFileConfig(mime_type=mime_type),
            )

        file_uri = getattr(uploaded, "uri", None)
        if not file_uri:
            raise ProviderGenerationError("File upload did not return a URI.")
        attachment.gemini_file_uri = file_uri
        return file_uri

    # ── transcription ────────────────────────────────────────────────────────

    async def transcribe_audio(
        self,
        model: str,
        audio_bytes: bytes,
        mime_type: str,
        prompt: str,
    ) -> tuple[str, int]:
        audio_b64 = base64.standard_b64encode(audio_bytes).decode("utf-8")
        try:
            response = await self.client.aio.models.generate_content(
                model=model,
                contents=[
                    {
                        "parts": [
                            {
                                "inline_data": {
                                    "mime_type": mime_type,
                                    "data": audio_b64,
                                }
                            },
                            {"text": prompt},
                        ]
                    }
                ],
            )
        except _GEMINI_ERROR_HANDLED as exc:
            raise self._wrap_error(exc) from exc

        transcript = (getattr(response, "text", None) or "").strip()
        tokens_used = 0
        if getattr(response, "usage_metadata", None):
            tokens_used = getattr(response.usage_metadata, "total_token_count", 0) or 0
        return transcript, tokens_used

    async def delete_file(self, file_uri: str) -> None:
        """Delete an uploaded file from Gemini File API."""
        try:
            name = file_uri.split("/")[-1]
            if not name.startswith("files/"):
                name = f"files/{name}"
            await self.client.aio.files.delete(name=name)
        except Exception as exc:
            logger.warning("Failed to delete gemini file %s: %s", file_uri, exc)
