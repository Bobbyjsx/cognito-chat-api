"""Tests for the Anthropic Claude provider abstraction."""

import uuid
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from anthropic import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    RateLimitError,
)

from app.models.attachments import AttachmentMetadata, AttachmentType
from app.providers.base import (
    ContentPart,
    GenerationConfig,
    ProviderAuthError,
    ProviderConnectionError,
    ProviderInvalidRequestError,
    ProviderModelNotFoundError,
    ProviderOverloadedError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnsupportedError,
)
from app.providers.claude import ClaudeProvider


def _build_mock_client():
    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock()
    return client


def _build_dummy_request():
    return httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def _build_dummy_response(status_code: int):
    return httpx.Response(status_code=status_code, request=_build_dummy_request())


# ── Generation & Usage Tests ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_claude_generate_text_and_usage():
    client = _build_mock_client()

    mock_text_block = MagicMock()
    mock_text_block.type = "text"
    mock_text_block.text = "Hello from Claude!"

    mock_response = MagicMock()
    mock_response.content = [mock_text_block]
    mock_response.usage = MagicMock(input_tokens=15, output_tokens=25)

    client.messages.create.return_value = mock_response
    provider = ClaudeProvider(api_key="sk-ant-test", client=client)

    result = await provider.generate(
        model="claude-3-7-sonnet",
        contents=[ContentPart(role="user", parts=[{"text": "Hello"}])],
    )

    assert result.text == "Hello from Claude!"
    assert result.total_tokens == 40
    assert result.tool_calls == []
    client.messages.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_claude_generate_with_tool_calls():
    client = _build_mock_client()

    mock_tool_block = MagicMock()
    mock_tool_block.type = "tool_use"
    mock_tool_block.id = "toolu_123"
    mock_tool_block.name = "get_weather"
    mock_tool_block.input = {"location": "San Francisco"}

    mock_response = MagicMock()
    mock_response.content = [mock_tool_block]
    mock_response.usage = MagicMock(input_tokens=20, output_tokens=10)

    client.messages.create.return_value = mock_response
    provider = ClaudeProvider(api_key="sk-ant-test", client=client)

    result = await provider.generate(
        model="claude-3-7-sonnet",
        contents=[ContentPart(role="user", parts=[{"text": "What is the weather in SF?"}])],
    )

    assert len(result.tool_calls) == 1
    tool_call = result.tool_calls[0]
    assert tool_call.id == "toolu_123"
    assert tool_call.name == "get_weather"
    assert tool_call.args == {"location": "San Francisco"}


# ── Streaming Tests ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_claude_stream_text_reasoning_and_usage():
    client = _build_mock_client()

    # Event 1: text delta
    event1 = MagicMock()
    event1.type = "content_block_delta"
    event1.delta = MagicMock()
    event1.delta.type = "text_delta"
    event1.delta.text = "Streaming "

    # Event 2: thinking delta (extended thinking)
    event2 = MagicMock()
    event2.type = "content_block_delta"
    event2.delta = MagicMock()
    event2.delta.type = "thinking_delta"
    event2.delta.thinking = "Thinking deeply..."

    # Event 3: text delta
    event3 = MagicMock()
    event3.type = "content_block_delta"
    event3.delta = MagicMock()
    event3.delta.type = "text_delta"
    event3.delta.text = "answer."

    class MockStreamContext:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        def __aiter__(self):
            return self._gen()

        async def _gen(self):
            for e in [event1, event2, event3]:
                yield e

        async def get_final_message(self):
            final_msg = MagicMock()
            final_msg.usage = MagicMock(input_tokens=10, output_tokens=20)
            return final_msg

    client.messages.stream = MagicMock(return_value=MockStreamContext())
    provider = ClaudeProvider(api_key="sk-ant-test", client=client)

    events = []
    async for event in provider.generate_stream(
        model="claude-3-7-sonnet",
        contents=[ContentPart(role="user", parts=[{"text": "Explain quantum physics"}])],
    ):
        events.append(event)

    assert [e.type for e in events] == ["text", "reasoning", "text", "usage"]
    assert events[0].token == "Streaming "
    assert events[1].token == "Thinking deeply..."
    assert events[2].token == "answer."
    assert events[3].total_tokens == 30


@pytest.mark.asyncio
async def test_claude_stream_tool_call():
    client = _build_mock_client()

    # Tool block start
    start_event = MagicMock()
    start_event.type = "content_block_start"
    mock_cb = MagicMock()
    mock_cb.type = "tool_use"
    mock_cb.id = "toolu_abc"
    mock_cb.name = "calculator"
    start_event.content_block = mock_cb

    # Tool block JSON deltas
    delta1 = MagicMock()
    delta1.type = "content_block_delta"
    delta1.delta = MagicMock(type="input_json_delta", partial_json='{"expression": ')

    delta2 = MagicMock()
    delta2.type = "content_block_delta"
    delta2.delta = MagicMock(type="input_json_delta", partial_json='"2 + 2"}')

    # Tool block stop
    stop_event = MagicMock()
    stop_event.type = "content_block_stop"

    class MockStreamContext:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        def __aiter__(self):
            return self._gen()

        async def _gen(self):
            for e in [start_event, delta1, delta2, stop_event]:
                yield e

        async def get_final_message(self):
            final_msg = MagicMock()
            final_msg.usage = MagicMock(input_tokens=5, output_tokens=15)
            return final_msg

    client.messages.stream = MagicMock(return_value=MockStreamContext())
    provider = ClaudeProvider(api_key="sk-ant-test", client=client)

    events = []
    async for event in provider.generate_stream(
        model="claude-3-7-sonnet",
        contents=[ContentPart(role="user", parts=[{"text": "Calculate 2+2"}])],
    ):
        events.append(event)

    assert len(events) == 2
    assert events[0].type == "tool_call"
    assert events[0].tool_call.id == "toolu_abc"
    assert events[0].tool_call.name == "calculator"
    assert events[0].tool_call.args == {"expression": "2 + 2"}
    assert events[1].type == "usage"
    assert events[1].total_tokens == 20


# ── Configuration & Translation Tests ────────────────────────────────────────


def test_claude_reasoning_config_translation():
    provider = ClaudeProvider(api_key="sk-ant-test")

    # 1. Fast / no thinking
    config_fast = GenerationConfig(thinking_budget=0)
    params_fast = provider._to_sdk_params(
        "claude-3-7-sonnet", [ContentPart(role="user", parts=[{"text": "hi"}])], config_fast
    )
    assert "thinking" not in params_fast
    assert params_fast["temperature"] == 0.7

    # 2. Balanced thinking (e.g. budget=4096)
    config_balanced = GenerationConfig(thinking_budget=4096)
    params_balanced = provider._to_sdk_params(
        "claude-3-7-sonnet", [ContentPart(role="user", parts=[{"text": "hi"}])], config_balanced
    )
    assert params_balanced["thinking"] == {"type": "enabled", "budget_tokens": 4096}
    assert params_balanced["max_tokens"] >= 4096
    assert "temperature" not in params_balanced  # temperature omitted when thinking is enabled

    # 3. Extended thinking (e.g. budget=16384)
    config_extended = GenerationConfig(thinking_budget=16384)
    params_extended = provider._to_sdk_params(
        "claude-3-7-sonnet", [ContentPart(role="user", parts=[{"text": "hi"}])], config_extended
    )
    assert params_extended["thinking"] == {"type": "enabled", "budget_tokens": 16384}
    assert params_extended["max_tokens"] > 16384


def test_claude_messages_conversion_alternation_and_system():
    provider = ClaudeProvider(api_key="sk-ant-test")

    contents = [
        ContentPart(role="user", parts=[{"text": "Turn 1"}]),
        ContentPart(role="model", parts=[{"text": "Turn 2"}]),
        ContentPart(role="user", parts=[{"text": "Turn 3"}]),
    ]
    config = GenerationConfig(system_instruction="You are a helpful assistant.")

    system, messages = provider._to_sdk_messages_and_system(contents, system_instruction=config.system_instruction)

    assert system == "You are a helpful assistant."
    assert len(messages) == 3
    assert messages[0]["role"] == "user"
    assert messages[0]["content"][0]["text"] == "Turn 1"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"][0]["text"] == "Turn 2"
    assert messages[2]["role"] == "user"
    assert messages[2]["content"][0]["text"] == "Turn 3"


def test_claude_tool_configs_conversion():
    provider = ClaudeProvider(api_key="sk-ant-test")
    tools = provider.build_tools(
        [
            {
                "kind": "function",
                "name": "lookup",
                "description": "Look up item",
                "schema": {"type": "object", "properties": {"query": {"type": "string"}}},
            }
        ]
    )
    assert len(tools) == 1
    assert tools[0]["name"] == "lookup"
    assert tools[0]["description"] == "Look up item"
    assert "properties" in tools[0]["input_schema"]


# ── Attachment Tests ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_claude_image_attachment():
    provider = ClaudeProvider(api_key="sk-ant-test")
    meta = AttachmentMetadata(
        user_id=uuid.uuid4(),
        filename="test.png",
        mime_type="image/png",
        type=AttachmentType.image,
    )
    parts = await provider.parts_for_attachment(meta, b"\x89PNG\r\n" + b"0" * 20)
    assert len(parts) == 1
    assert "inline_data" in parts[0]
    assert parts[0]["inline_data"]["mime_type"] == "image/png"


@pytest.mark.asyncio
async def test_claude_pdf_attachment():
    provider = ClaudeProvider(api_key="sk-ant-test")
    meta = AttachmentMetadata(
        user_id=uuid.uuid4(),
        filename="doc.pdf",
        mime_type="application/pdf",
        type=AttachmentType.pdf,
    )
    parts = await provider.parts_for_attachment(meta, b"%PDF-1.4" + b"0" * 20)
    assert len(parts) == 1
    assert parts[0]["inline_data"]["mime_type"] == "application/pdf"


@pytest.mark.asyncio
async def test_claude_transcribe_audio_raises_unsupported():
    provider = ClaudeProvider(api_key="sk-ant-test")
    with pytest.raises(ProviderUnsupportedError):
        await provider.transcribe_audio("claude-3-7-sonnet", b"\x00", "audio/wav", "transcribe")


# ── Error Normalization Tests ────────────────────────────────────────────────


def test_claude_error_normalization():
    provider = ClaudeProvider(api_key="sk-ant-test")
    req = _build_dummy_request()
    resp404 = _build_dummy_response(404)
    resp429 = _build_dummy_response(429)
    resp401 = _build_dummy_response(401)
    resp400 = _build_dummy_response(400)
    resp500 = _build_dummy_response(500)

    # 404
    err_404 = NotFoundError("model not found", response=resp404, body=None)
    assert isinstance(provider.normalize_error(err_404), ProviderModelNotFoundError)

    # 429
    err_429 = RateLimitError("rate limited", response=resp429, body=None)
    assert isinstance(provider.normalize_error(err_429), ProviderRateLimitError)

    # 401
    err_401 = AuthenticationError("bad api key", response=resp401, body=None)
    assert isinstance(provider.normalize_error(err_401), ProviderAuthError)

    # 400
    err_400 = BadRequestError("invalid prompt", response=resp400, body=None)
    assert isinstance(provider.normalize_error(err_400), ProviderInvalidRequestError)

    # 500 / Overloaded
    err_500 = InternalServerError("servers busy", response=resp500, body=None)
    assert isinstance(provider.normalize_error(err_500), ProviderOverloadedError)

    # Connection / Timeout
    err_conn = APIConnectionError(request=req)
    assert isinstance(provider.normalize_error(err_conn), ProviderConnectionError)

    err_time = APITimeoutError(request=req)
    assert isinstance(provider.normalize_error(err_time), ProviderTimeoutError)
