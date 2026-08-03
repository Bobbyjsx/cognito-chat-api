"""Tests for the Gemini provider abstraction."""

import base64
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models.attachments import AttachmentMetadata, AttachmentType
from app.providers.base import (
    ContentPart,
    ProviderGenerationError,
    ProviderModelNotFoundError,
    classify_provider_error,
)
from app.providers.gemini import GeminiProvider


def _client_with_aio():
    """Build a client mock with pinned aio sub-mocks (avoiding the transient
    MagicMock attribute pitfall)."""
    client = MagicMock()
    client.aio = MagicMock()
    client.aio.models = MagicMock()
    client.aio.files = MagicMock()
    return client


def _chunk_with_parts(parts, usage=None):
    content = MagicMock(parts=parts)
    candidate = MagicMock(content=content)
    chunk = MagicMock(candidates=[candidate])
    chunk.usage_metadata = MagicMock(total_token_count=usage) if usage else MagicMock(total_token_count=0)
    return chunk


def _text_part(text):
    return MagicMock(
        text=text,
        thought=None,
        function_call=None,
        executable_code=None,
        code_execution_result=None,
    )


def test_build_tools_maps_all_kinds():
    provider = GeminiProvider(api_key="x")
    tools = provider.build_tools(
        [
            {"kind": "code_execution"},
            {"kind": "google_search"},
            {"kind": "function", "name": "add", "description": "Adds", "schema": {"type": "object"}},
        ]
    )
    assert tools[0].code_execution is not None
    assert tools[1].google_search is not None
    assert tools[2].function_declarations[0].name == "add"
    assert tools[2].function_declarations[0].description == "Adds"


def test_build_tools_empty_when_no_configs():
    provider = GeminiProvider(api_key="x")
    assert provider.build_tools([]) == []


def test_to_sdk_contents_converts_shapes():
    provider = GeminiProvider(api_key="x")
    sdk = provider._to_sdk_contents(
        [
            ContentPart(
                role="user",
                parts=[
                    {"text": "hello"},
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": base64.standard_b64encode(b"\x89PNG").decode(),
                        }
                    },
                ],
            )
        ]
    )
    assert sdk[0].role == "user"
    assert sdk[0].parts[0].text == "hello"
    assert sdk[0].parts[1].inline_data.mime_type == "image/png"
    assert sdk[0].parts[1].inline_data.data == b"\x89PNG"


def test_stream_events_parse_text_thought_and_function_calls():
    provider = GeminiProvider(api_key="x")
    parts = [
        _text_part("visible"),
        MagicMock(
            text="thinking...",
            thought=True,
            function_call=None,
            executable_code=None,
            code_execution_result=None,
        ),
        MagicMock(
            text=None,
            thought=None,
            function_call=MagicMock(id="fc_1", args={"a": 1}),
            executable_code=None,
            code_execution_result=None,
        ),
    ]
    parts[2].function_call.name = "add"
    events = provider._extract_stream_events(_chunk_with_parts(parts, usage=7), search_announced=False)

    assert [e.type for e in events] == ["text", "reasoning", "tool_call", "usage"]
    assert events[0].token == "visible"
    assert events[1].token == "thinking..."
    assert events[2].tool_call.name == "add"
    assert events[2].tool_call.kind == "function"
    assert events[3].total_tokens == 7


def test_stream_events_parse_code_execution():
    provider = GeminiProvider(api_key="x")
    exec_part = MagicMock(
        text=None,
        thought=None,
        function_call=None,
        executable_code=MagicMock(language="python", code="print(1)"),
        code_execution_result=None,
    )
    result_part = MagicMock(
        text=None,
        thought=None,
        function_call=None,
        executable_code=None,
        code_execution_result=MagicMock(outcome="OUTCOME_OK", output="1"),
    )
    events = provider._extract_stream_events(_chunk_with_parts([exec_part]), search_announced=False)
    events += provider._extract_stream_events(_chunk_with_parts([result_part]), search_announced=False)

    assert events[0].type == "tool_call"
    assert events[0].tool_call.name == "code_execution"
    assert events[0].tool_call.kind == "server"
    assert events[1].type == "usage"
    assert events[2].type == "tool_result"
    assert events[2].tool_result.output["output"] == "1"


def test_grounding_synthesizes_search_events():
    provider = GeminiProvider(api_key="x")
    chunk = MagicMock(candidates=[])
    chunk.usage_metadata = MagicMock(total_token_count=0)
    chunk.grounding_metadata = MagicMock(
        grounding_chunks=[MagicMock(web=MagicMock(title="Example", uri="https://example.com"))]
    )
    events = provider._extract_grounding_events(chunk, search_announced=False)
    assert events[0].type == "tool_call"
    assert events[0].tool_call.name == "google_search"
    assert events[1].type == "tool_result"
    assert events[1].tool_result.output["sources"][0]["title"] == "Example"


@pytest.mark.asyncio
async def test_image_attachment_uses_inline_data():
    provider = GeminiProvider(api_key="x")
    metadata = AttachmentMetadata(
        user_id=uuid.uuid4(), filename="photo.png", mime_type="image/png", type=AttachmentType.image
    )
    parts = await provider.parts_for_attachment(metadata, b"\x89PNG\r\n\x1a\n" + b"0" * 100)
    assert "inline_data" in parts[0]
    assert parts[0]["inline_data"]["mime_type"] == "image/png"


@pytest.mark.asyncio
async def test_video_attachment_uploads_to_files_api_and_caches_uri():
    client = _client_with_aio()
    client.aio.files.upload = AsyncMock(return_value=MagicMock(uri="files/video-123"))
    provider = GeminiProvider(api_key="x", client=client)

    metadata = AttachmentMetadata(
        user_id=uuid.uuid4(), filename="clip.mp4", mime_type="video/mp4", type=AttachmentType.video
    )
    parts = await provider.parts_for_attachment(metadata, b"\x00" * 100)
    assert parts[0] == {"file_data": {"file_uri": "files/video-123", "mime_type": "video/mp4"}}
    assert metadata.gemini_file_uri == "files/video-123"
    client.aio.files.upload.assert_awaited_once()

    # Second call reuses the cached URI without re-uploading
    parts = await provider.parts_for_attachment(metadata, b"\x00" * 100)
    assert parts[0]["file_data"]["file_uri"] == "files/video-123"
    assert client.aio.files.upload.await_count == 1


@pytest.mark.asyncio
async def test_generate_wraps_sdk_response():
    client = _client_with_aio()
    response = MagicMock(
        text="hello world",
        usage_metadata=MagicMock(total_token_count=42),
        candidates=[],
    )
    client.aio.models.generate_content = AsyncMock(return_value=response)
    provider = GeminiProvider(api_key="x", client=client)

    result = await provider.generate("gemini-x", [ContentPart(role="user", parts=[{"text": "hi"}])])
    assert result.text == "hello world"
    assert result.total_tokens == 42


@pytest.mark.asyncio
async def test_transcribe_audio_delegates():
    client = _client_with_aio()
    response = MagicMock(text="  transcribed  ", usage_metadata=MagicMock(total_token_count=9), candidates=[])
    client.aio.models.generate_content = AsyncMock(return_value=response)
    provider = GeminiProvider(api_key="x", client=client)

    transcript, tokens = await provider.transcribe_audio("gemini-y", b"\x00" * 10, "audio/webm", "prompt")
    assert transcript == "transcribed"
    assert tokens == 9


@pytest.mark.asyncio
async def test_error_mapping_404_is_model_not_found():
    from google.genai import errors as genai_errors

    error = genai_errors.ClientError(
        code=404,
        response_json={"error": {"code": 404, "status": "NOT_FOUND", "message": "models/gemini-x not found"}},
    )
    client = _client_with_aio()
    client.aio.models.generate_content = AsyncMock(side_effect=error)
    provider = GeminiProvider(api_key="x", client=client)

    with pytest.raises(ProviderModelNotFoundError):
        await provider.generate("gemini-x", [])


def test_classify_provider_error_statuses():
    status, code, _ = classify_provider_error(ProviderModelNotFoundError("nope"))
    assert (status, code) == (404, "MODEL_NOT_FOUND")

    status, code, _ = classify_provider_error(ProviderGenerationError("boom", status_code=502))
    assert (status, code) == (502, "GENERATION_FAILED")

    status, code, _ = classify_provider_error(RuntimeError("surprise"))
    assert (status, code) == (500, "GENERATION_FAILED")
