"""Integration tests for Smart Model Router within the application."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.ai.router import (
    CompositeRequestAnalyzer,
    HeuristicFallbackAnalyzer,
    NoEligibleModelsError,
    RequestContext,
    RoutingMode,
    RoutingPolicy,
    SmartModelRouter,
    TaskType,
)
from app.models.attachments import AttachmentType
from app.models.config import AppConfigDB


@pytest.mark.asyncio
async def test_router_end_to_end_simple_query():
    router = SmartModelRouter(analyzer=HeuristicFallbackAnalyzer())
    decision = await router.route("Hello, how are you today?")
    assert decision.selected_model_id in ("gemini-3.1-flash-lite", "gemini-3.5-flash-lite")
    assert decision.task_type == TaskType.CONVERSATION
    assert decision.complexity <= 0.3
    assert len(decision.fallback_models) > 0


@pytest.mark.asyncio
async def test_router_end_to_end_complex_coding():
    router = SmartModelRouter(analyzer=HeuristicFallbackAnalyzer())
    decision = await router.route(
        "Can you refactor this complex microservice architecture and write the python async code? ```def process(): pass```",
        policy=RoutingMode.EXTENDED,
    )
    assert decision.selected_model_id in ("gemini-3.6-flash", "gemini-3.1-pro-preview")
    assert decision.task_type == TaskType.CODING
    assert decision.complexity >= 0.5


@pytest.mark.asyncio
async def test_router_end_to_end_vision_filtering():
    router = SmartModelRouter(analyzer=HeuristicFallbackAnalyzer())
    ctx = RequestContext(has_attachments=True, attachment_types=[AttachmentType.image])
    decision = await router.route("Analyze the visual layout of this screenshot.", context=ctx)
    assert decision.analysis.vision_required is True
    # The chosen model must support vision
    config = AppConfigDB()
    selected_cfg = config.models_list[decision.selected_model_id]
    assert selected_cfg.supports_vision is True


@pytest.mark.asyncio
async def test_router_user_plan_constraint():
    router = SmartModelRouter(analyzer=HeuristicFallbackAnalyzer())
    policy = RoutingPolicy(allowed_models=["gemini-3.5-flash-lite"])
    decision = await router.route("Refactor this whole software system.", policy=policy)
    # Even though request is complex, policy restricts to flash-lite
    assert decision.selected_model_id == "gemini-3.5-flash-lite"


@pytest.mark.asyncio
async def test_router_no_eligible_models_raises_error():
    router = SmartModelRouter(analyzer=HeuristicFallbackAnalyzer())
    # Restrict allowed models to empty list
    policy = RoutingPolicy(allowed_models=[])
    with pytest.raises(NoEligibleModelsError):
        await router.route("Hello", policy=policy)


@pytest.mark.asyncio
async def test_router_or_default_fallback():
    router = SmartModelRouter(analyzer=HeuristicFallbackAnalyzer())
    cfg = AppConfigDB()
    # If explicit model requested, route_or_default returns it directly
    selected, _fallbacks, decision = await router.route_or_default(
        message="Hello",
        requested_model="gemini-3.5-flash",
        config=cfg,
    )
    assert selected == "gemini-3.5-flash"
    assert decision is None


@pytest.mark.asyncio
async def test_router_analyzer_failure_graceful_fallback():
    failing_analyzer = MagicMock()
    failing_analyzer.analyze = AsyncMock(side_effect=RuntimeError("Primary analyzer network failure"))
    composite = CompositeRequestAnalyzer(
        primary_analyzer=failing_analyzer,
        fallback_analyzer=HeuristicFallbackAnalyzer(),
    )
    router = SmartModelRouter(analyzer=composite)
    decision = await router.route("Solve 2x + 4 = 10")
    # Must successfully complete routing via heuristic fallback
    assert decision.selected_model_id is not None
    assert decision.task_type == TaskType.MATH_REASONING


def _get_auth_headers(client: TestClient, email: str = "routeruser@example.com") -> dict:
    client.post("/auth/signup", json={"email": email, "password": "securepassword123"})
    login_resp = client.post("/auth/login", json={"email": email, "password": "securepassword123"})
    token = login_resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_http_chat_with_auto_smart_routing(client: TestClient, mock_agent):
    headers = _get_auth_headers(client, "user_auto@example.com")
    payload = {
        "message": "Hi, tell me a quick joke",
    }
    response = client.post("/agent/chat", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert data["model"] in ("gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-3.6-flash")


def test_http_chat_with_fast_routing_mode(client: TestClient, mock_agent):
    headers = _get_auth_headers(client, "user_fast@example.com")
    payload = {
        "message": "What is 2 + 2?",
        "routing_mode": "fast",
    }
    response = client.post("/agent/chat", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "flash-lite" in data["model"]


def test_http_chat_with_explicit_model_override(client: TestClient, mock_agent):
    headers = _get_auth_headers(client, "user_explicit@example.com")
    payload = {
        "message": "Hello from manual selection",
        "model": "gemini-3.5-flash",
        "reasoning": "balanced",
    }
    response = client.post("/agent/chat", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "gemini-3.5-flash"
    assert data["reasoning"] == "balanced"


def test_http_chat_with_auto_model_uses_reasoning_as_policy(client: TestClient, mock_agent):
    headers = _get_auth_headers(client, "user_auto_reasoning@example.com")
    payload = {
        "message": "What is 2 + 2?",
        "model": "auto",
        "reasoning": "fast",
        # Note: no routing_mode needed!
    }
    response = client.post("/agent/chat", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "flash-lite" in data["model"]
    assert data["reasoning"] == "fast"


def test_http_chat_with_explicit_claude_uses_reasoning_as_effort(client: TestClient, mock_agent):
    from unittest.mock import AsyncMock, patch

    from app.providers.base import GenerationResult

    headers = _get_auth_headers(client, "user_claude_effort@example.com")
    payload = {
        "message": "Solve complex architecture task",
        "model": "claude-sonnet-4-5",
        "reasoning": "extended",
        # Note: no routing_mode needed!
    }
    with patch(
        "app.providers.claude.ClaudeProvider.generate",
        new_callable=AsyncMock,
        return_value=GenerationResult(text="Claude answer", total_tokens=100),
    ):
        response = client.post("/agent/chat", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "claude-sonnet-4-5"
    assert data["reasoning"] == "extended"
