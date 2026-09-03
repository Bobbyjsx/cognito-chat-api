"""Integration tests for Multi-Provider model architecture across Router, Registry, and Providers."""

import uuid
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from anthropic import RateLimitError

from app.ai.router import HeuristicFallbackAnalyzer, RoutingMode, SmartModelRouter
from app.models.chats import ChatResponse
from app.models.config import AppConfigDB, ModelProvider
from app.models.users import UserDB
from app.providers.base import (
    GenerationEvent,
    GenerationResult,
    ProviderInvalidRequestError,
)
from app.providers.claude import ClaudeProvider
from app.providers.gemini import GeminiProvider
from app.providers.registry import ProviderRegistry
from app.repositories.chats import ChatRepository
from app.repositories.config import ConfigRepository
from app.repositories.users import UserRepository
from app.services.attachments import AttachmentService
from app.services.chats import AgentService
from app.tools.base import BaseTool, ToolOutput
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry


class EchoTool(BaseTool):
    name = "echo_tool"
    description = "Echoes back input"

    @property
    def schema(self):
        return {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }

    async def execute(self, args):
        return ToolOutput(content=f"Echo: {args.get('text', '')}")


@pytest.fixture
def mock_db():
    db = MagicMock()
    return db


@pytest.fixture
def multi_provider_setup():
    # 1. Mock Gemini Provider
    gemini = MagicMock(spec=GeminiProvider)
    gemini.name = "gemini"
    gemini.is_retryable_error = GeminiProvider.is_retryable_error
    gemini.supports = lambda cap: True
    gemini.supports_model = lambda m: m.startswith("gemini")
    gemini.parts_for_attachment = AsyncMock(return_value=[{"inline_data": {"mime_type": "image/png", "data": "abc"}}])
    gemini.generate = AsyncMock(
        return_value=GenerationResult(text="Response from Gemini", total_tokens=25, tool_calls=[])
    )

    async def gemini_stream(model, contents, config=None):
        yield GenerationEvent(type="text", token="Response ")
        yield GenerationEvent(type="text", token="from Gemini")
        yield GenerationEvent(type="usage", total_tokens=25)

    gemini.generate_stream = MagicMock(side_effect=gemini_stream)

    # 2. Mock Claude Provider
    claude = MagicMock(spec=ClaudeProvider)
    claude.name = "anthropic"
    claude.is_retryable_error = ClaudeProvider.is_retryable_error
    claude.supports = lambda cap: cap not in ("audio", "audio_transcription", "google_search")
    claude.supports_model = lambda m: m.startswith("claude")
    claude.parts_for_attachment = AsyncMock(return_value=[{"inline_data": {"mime_type": "image/png", "data": "abc"}}])
    claude.generate = AsyncMock(
        return_value=GenerationResult(text="Response from Claude 3.7", total_tokens=50, tool_calls=[])
    )

    async def claude_stream(model, contents, config=None):
        yield GenerationEvent(type="reasoning", token="Analyzing with extended thinking...")
        yield GenerationEvent(type="text", token="Response from Claude 3.7")
        yield GenerationEvent(type="usage", total_tokens=50)

    claude.generate_stream = MagicMock(side_effect=claude_stream)

    # 3. Provider Registry
    registry = ProviderRegistry()
    registry.register("gemini", gemini)
    registry.register("google", gemini)
    registry.register(ModelProvider.GOOGLE, gemini)
    registry.register("anthropic", claude)
    registry.register("claude", claude)
    registry.register(ModelProvider.ANTHROPIC, claude)

    # 4. Tool Registry
    tool_registry = ToolRegistry()
    tool_registry.register(EchoTool())

    executor = ToolExecutor(registry=tool_registry, provider=registry)
    return {
        "gemini": gemini,
        "claude": claude,
        "provider_registry": registry,
        "tool_registry": tool_registry,
        "executor": executor,
    }


@pytest.mark.asyncio
async def test_agent_service_explicit_claude_model(multi_provider_setup):
    setup = multi_provider_setup
    chat_repo = MagicMock(spec=ChatRepository)
    session_mock = MagicMock(id=uuid.uuid4(), title="Test Session", messages=[])
    chat_repo.get_session = AsyncMock(return_value=(session_mock, False))
    chat_repo.add_message = AsyncMock()

    user_repo = MagicMock(spec=UserRepository)
    user_repo.atomic_increment_if_within_limit = AsyncMock(return_value=True)

    config_repo = MagicMock(spec=ConfigRepository)
    config = AppConfigDB()
    config_repo.get_config = AsyncMock(return_value=config)

    storage = MagicMock()
    attachment_service = AttachmentService(MagicMock(), storage, setup["gemini"])

    service = AgentService(
        chat_repo=chat_repo,
        user_repo=user_repo,
        config_repo=config_repo,
        attachment_service=attachment_service,
        provider_registry=setup["provider_registry"],
        registry=setup["tool_registry"],
        executor=setup["executor"],
    )

    user = UserDB(id=uuid.uuid4(), email="claude_user@example.com", hashed_password="")

    # Process chat requesting Claude
    response = await service.process_chat(
        user=user,
        message_text="Write a complex distributed algorithm in Rust",
        session_id=session_mock.id,
        requested_model="claude-3-7-sonnet",
        requested_reasoning="extended",
    )

    assert isinstance(response, ChatResponse)
    assert response.model == "claude-3-7-sonnet"
    assert response.response == "Response from Claude 3.7"
    setup["claude"].generate.assert_awaited_once()
    setup["gemini"].generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_agent_service_claude_streaming(multi_provider_setup):
    setup = multi_provider_setup
    chat_repo = MagicMock(spec=ChatRepository)
    session_mock = MagicMock(id=uuid.uuid4(), title="Stream Session", messages=[])
    chat_repo.get_session = AsyncMock(return_value=(session_mock, False))
    chat_repo.add_message = AsyncMock()

    user_repo = MagicMock(spec=UserRepository)
    user_repo.atomic_increment_if_within_limit = AsyncMock(return_value=True)

    config_repo = MagicMock(spec=ConfigRepository)
    config = AppConfigDB()
    config_repo.get_config = AsyncMock(return_value=config)

    storage = MagicMock()
    attachment_service = AttachmentService(MagicMock(), storage, setup["gemini"])

    service = AgentService(
        chat_repo=chat_repo,
        user_repo=user_repo,
        config_repo=config_repo,
        attachment_service=attachment_service,
        provider_registry=setup["provider_registry"],
        registry=setup["tool_registry"],
        executor=setup["executor"],
    )

    user = UserDB(id=uuid.uuid4(), email="stream_user@example.com", hashed_password="")

    chunks = []
    async for chunk in service.stream_chat(
        user=user,
        message_text="Analyze architecture",
        session_id=session_mock.id,
        requested_model="claude-3-7-sonnet",
    ):
        chunks.append(chunk)

    full_stream = "".join(chunks)
    assert "event: session" in full_stream
    assert "event: chunk" in full_stream
    assert "event: done" in full_stream
    assert "Analyzing with extended thinking..." in full_stream
    assert "Response from Claude 3.7" in full_stream


@pytest.mark.asyncio
async def test_cross_provider_fallback_on_rate_limit(multi_provider_setup):
    setup = multi_provider_setup

    # Claude fails with RateLimitError (429)
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    resp = httpx.Response(status_code=429, request=req)
    claude_err = RateLimitError("rate limited", response=resp, body=None)
    setup["claude"].generate = AsyncMock(side_effect=ClaudeProvider(api_key="x").normalize_error(claude_err))

    chat_repo = MagicMock(spec=ChatRepository)
    session_mock = MagicMock(id=uuid.uuid4(), title="Fallback Session", messages=[])
    chat_repo.get_session = AsyncMock(return_value=(session_mock, False))
    chat_repo.add_message = AsyncMock()

    user_repo = MagicMock(spec=UserRepository)
    user_repo.atomic_increment_if_within_limit = AsyncMock(return_value=True)

    config_repo = MagicMock(spec=ConfigRepository)
    config = AppConfigDB()
    config_repo.get_config = AsyncMock(return_value=config)

    storage = MagicMock()
    attachment_service = AttachmentService(MagicMock(), storage, setup["gemini"])

    service = AgentService(
        chat_repo=chat_repo,
        user_repo=user_repo,
        config_repo=config_repo,
        attachment_service=attachment_service,
        provider_registry=setup["provider_registry"],
        registry=setup["tool_registry"],
        executor=setup["executor"],
    )

    user = UserDB(id=uuid.uuid4(), email="fallback_user@example.com", hashed_password="")

    # Claude was primary, but fails with rate limit -> should fallback to next model (Gemini)
    response = await service.process_chat(
        user=user,
        message_text="Hello",
        session_id=session_mock.id,
        requested_model="claude-3-7-sonnet",
    )

    assert response.response == "Response from Gemini"
    setup["claude"].generate.assert_awaited_once()
    setup["gemini"].generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_non_retryable_error_bypasses_fallback(multi_provider_setup):
    setup = multi_provider_setup

    # Claude fails with non-retryable 400 error
    setup["claude"].generate = AsyncMock(side_effect=ProviderInvalidRequestError("Bad prompt syntax", status_code=400))

    chat_repo = MagicMock(spec=ChatRepository)
    session_mock = MagicMock(id=uuid.uuid4(), title="NonRetry Session", messages=[])
    chat_repo.get_session = AsyncMock(return_value=(session_mock, False))
    chat_repo.add_message = AsyncMock()

    user_repo = MagicMock(spec=UserRepository)
    user_repo.atomic_increment_if_within_limit = AsyncMock(return_value=True)

    config_repo = MagicMock(spec=ConfigRepository)
    config = AppConfigDB()
    config_repo.get_config = AsyncMock(return_value=config)

    storage = MagicMock()
    attachment_service = AttachmentService(MagicMock(), storage, setup["gemini"])

    service = AgentService(
        chat_repo=chat_repo,
        user_repo=user_repo,
        config_repo=config_repo,
        attachment_service=attachment_service,
        provider_registry=setup["provider_registry"],
        registry=setup["tool_registry"],
        executor=setup["executor"],
    )

    user = UserDB(id=uuid.uuid4(), email="nonretry_user@example.com", hashed_password="")

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        await service.process_chat(
            user=user,
            message_text="Hello",
            session_id=session_mock.id,
            requested_model="claude-3-7-sonnet",
        )

    assert exc_info.value.status_code == 400
    # Gemini should NOT have been called
    setup["gemini"].generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_router_selects_claude_for_high_complexity_coding():
    router = SmartModelRouter(analyzer=HeuristicFallbackAnalyzer())
    config = AppConfigDB()

    decision = await router.route(
        message="Please design and write a highly concurrent async distributed lock manager with Redis and Raft consensus in Rust. ```fn lock() {}```",
        policy=RoutingMode.EXTENDED,
        config=config,
    )

    assert decision.selected_model_id in (
        "claude-3-7-sonnet",
        "claude-3-5-sonnet",
        "gemini-3.1-pro-preview",
        "gemini-3.6-flash",
    )
    assert decision.provider in (ModelProvider.ANTHROPIC, ModelProvider.GOOGLE)


def test_create_default_provider_registry_vertex_claude():
    from app.core.config import Settings
    from app.providers.registry import create_default_provider_registry

    settings = Settings(
        claude_backend="vertex",
        anthropic_vertex_project_id="test-vertex-project",
        anthropic_vertex_region="us-east5",
    )

    registry = create_default_provider_registry(settings)
    assert "anthropic" in registry._providers
    assert "vertex_claude" in registry._providers

    claude_provider = registry.get("anthropic")
    assert isinstance(claude_provider, ClaudeProvider)
    assert claude_provider.is_vertex is True
    assert claude_provider.project_id == "test-vertex-project"
    assert claude_provider.region == "us-east5"
