from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models.config import AppConfigDB, ReasoningLevel, ToolName
from app.providers.base import GenerationEvent, ToolCall, ToolResult
from app.providers.claude import ClaudeProvider
from app.providers.gemini import GeminiProvider
from app.services.chats import AgentService
from app.tools.registry import ToolRegistry
from app.utils.prompts import get_base_system_instructions


def test_get_base_system_instructions_includes_current_date():
    instructions = get_base_system_instructions()
    now_year = str(datetime.now(timezone.utc).year)
    assert now_year in instructions
    assert "Cognito" in instructions
    assert "Today's date is" in instructions


def test_gemini_extract_grounding_events_from_candidates():
    provider = GeminiProvider(api_key="mock")
    cand = MagicMock()
    cand.grounding_metadata = MagicMock(
        web_search_queries=["current stock price"],
        grounding_chunks=[MagicMock(web=MagicMock(title="Yahoo Finance", uri="https://finance.yahoo.com"))],
    )
    chunk = MagicMock(candidates=[cand], usage_metadata=None)
    chunk.grounding_metadata = None

    events = provider._extract_grounding_events(chunk, search_announced=False, search_call_id="call_test_123")
    assert len(events) == 2
    assert events[0].type == "tool_call"
    assert events[0].tool_call.id == "call_test_123"
    assert events[0].tool_call.name == "google_search"
    assert events[0].tool_call.args == {"query": "current stock price"}

    assert events[1].type == "tool_result"
    assert events[1].tool_result.id == "call_test_123"
    assert events[1].tool_result.name == "google_search"
    assert events[1].tool_result.output["sources"][0]["title"] == "Yahoo Finance"


def test_claude_build_tools_maps_google_search():
    provider = ClaudeProvider(api_key="mock")
    tool_configs = [{"kind": "google_search"}]
    tools = provider.build_tools(tool_configs)
    assert len(tools) == 1
    assert tools[0]["name"] == "google_search"
    assert "query" in tools[0]["input_schema"]["properties"]


def test_agent_service_serialize_tool_events():
    call_ev = GenerationEvent(
        type="tool_call",
        tool_call=ToolCall(id="call_abc", name="google_search", args={"query": "test"}),
    )
    serialized_call = AgentService._serialize_event(call_ev)
    assert serialized_call["type"] == "tool_call"
    assert serialized_call["tool_name"] == "google_search"
    assert serialized_call["tool_call_id"] == "call_abc"
    assert serialized_call["args"] == {"query": "test"}
    assert serialized_call["input"] == {"query": "test"}

    result_ev = GenerationEvent(
        type="tool_result",
        tool_result=ToolResult(id="call_abc", name="google_search", output={"sources": []}),
    )
    serialized_res = AgentService._serialize_event(result_ev)
    assert serialized_res["type"] == "tool_result"
    assert serialized_res["tool_name"] == "google_search"
    assert serialized_res["tool_call_id"] == "call_abc"
    assert serialized_res["output"] == {"sources": []}


@pytest.mark.asyncio
async def test_dynamic_tool_attachment_in_validate_and_resolve_config():
    config_repo = MagicMock()
    mock_config = AppConfigDB(
        allowed_tools=[ToolName.GOOGLE_SEARCH, ToolName.CODE_EXECUTION],
        allowed_reasoning_levels=[ReasoningLevel.FAST],
        default_reasoning_level=ReasoningLevel.FAST,
        default_text_model="gemini-3.6-flash",
    )
    config_repo.get_config = AsyncMock(return_value=mock_config)

    registry = ToolRegistry()
    registry.register_defaults()

    mock_router = MagicMock()
    mock_analyzer = MagicMock()

    analysis_simple = MagicMock(coding_required=0.0, task_type=MagicMock(value="conversation"), web_required=False)
    analysis_web = MagicMock(coding_required=0.0, task_type=MagicMock(value="information"), web_required=True)

    mock_analyzer.analyze = AsyncMock(side_effect=[analysis_simple, analysis_web])
    mock_router.analyzer = mock_analyzer

    service = AgentService(
        chat_repo=MagicMock(),
        user_repo=MagicMock(),
        config_repo=config_repo,
        attachment_service=None,
        provider=MagicMock(),
        registry=registry,
        router=mock_router,
    )

    _, _, _, tools_simple, _ = await service.validate_and_resolve_config(
        requested_model="gemini-3.6-flash",
        message_text="Write a poem about rain",
    )
    assert tools_simple == []

    _, _, _, tools_web, _ = await service.validate_and_resolve_config(
        requested_model="gemini-3.6-flash",
        message_text="What is the stock price of Apple right now?",
    )
    assert len(tools_web) == 1
    assert tools_web[0]["kind"] == "google_search"
