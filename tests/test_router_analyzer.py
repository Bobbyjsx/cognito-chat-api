"""Unit tests for Request Analyzer implementations."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.ai.router.analyzer import (
    CompositeRequestAnalyzer,
    GeminiFlashLiteAnalyzer,
    HeuristicFallbackAnalyzer,
)
from app.ai.router.exceptions import AnalyzerError
from app.ai.router.schemas import RequestContext, TaskType


@pytest.mark.asyncio
async def test_heuristic_analyzer_simple_greeting():
    analyzer = HeuristicFallbackAnalyzer()
    res = await analyzer.analyze("hello there, how are you?")
    assert res.task_type == TaskType.CONVERSATION
    assert res.complexity <= 0.3
    assert res.coding_required == 0.0
    assert res.vision_required is False


@pytest.mark.asyncio
async def test_heuristic_analyzer_coding_request():
    analyzer = HeuristicFallbackAnalyzer()
    res = await analyzer.analyze("Can you write a python function to refactor this class: def solve(): pass")
    assert res.task_type == TaskType.CODING
    assert res.coding_required >= 0.7
    assert res.complexity >= 0.5


@pytest.mark.asyncio
async def test_heuristic_analyzer_math_reasoning():
    analyzer = HeuristicFallbackAnalyzer()
    res = await analyzer.analyze("Solve this equation and prove the theorem: x^2 + 5x + 6 = 0")
    assert res.task_type == TaskType.MATH_REASONING
    assert res.reasoning_required >= 0.75
    assert res.complexity >= 0.7


@pytest.mark.asyncio
async def test_heuristic_analyzer_creative_writing():
    analyzer = HeuristicFallbackAnalyzer()
    res = await analyzer.analyze("Write a story about a spaceship exploring an alien planet.")
    assert res.task_type == TaskType.CREATIVE_WRITING
    assert res.creative_required >= 0.8


@pytest.mark.asyncio
async def test_heuristic_analyzer_attachments_vision():
    from app.models.attachments import AttachmentType

    analyzer = HeuristicFallbackAnalyzer()
    ctx = RequestContext(has_attachments=True, attachment_types=[AttachmentType.image])
    res = await analyzer.analyze("What is in this image?", context=ctx)
    assert res.vision_required is True
    assert res.context_required >= 0.5


@pytest.mark.asyncio
async def test_gemini_flash_lite_analyzer_success():
    analyzer = GeminiFlashLiteAnalyzer(api_key="test-key")
    mock_response = MagicMock()
    mock_response.text = (
        '{"task_type": "coding", "complexity": 0.85, "reasoning_required": 0.8, '
        '"context_required": 0.3, "coding_required": 0.95, "creative_required": 0.0, '
        '"vision_required": false, "web_required": false, "tool_calling_required": true, '
        '"structured_output_required": false}'
    )

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    with patch.object(analyzer, "_get_client", return_value=mock_client):
        result = await analyzer.analyze("Refactor my async database connector.")
        assert result.task_type == TaskType.CODING
        assert result.complexity == 0.85
        assert result.coding_required == 0.95


@pytest.mark.asyncio
async def test_gemini_flash_lite_analyzer_error_handling():
    analyzer = GeminiFlashLiteAnalyzer(api_key="test-key")
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(side_effect=RuntimeError("API quota exceeded"))

    with patch.object(analyzer, "_get_client", return_value=mock_client), pytest.raises(AnalyzerError):
        await analyzer.analyze("Hello")


@pytest.mark.asyncio
async def test_composite_analyzer_fallback():
    mock_primary = MagicMock()
    mock_primary.analyze = AsyncMock(side_effect=RuntimeError("Timeout"))

    composite = CompositeRequestAnalyzer(primary_analyzer=mock_primary, prefer_heuristic=False)
    res = await composite.analyze("write a python script to parse logs")
    # Must seamlessly fallback to heuristic analyzer without throwing
    assert res.task_type == TaskType.CODING
    assert res.coding_required > 0.0
    mock_primary.analyze.assert_called_once()


@pytest.mark.asyncio
async def test_heuristic_analyzer_date_queries():
    analyzer = HeuristicFallbackAnalyzer()
    date_queries = [
        "What's the date?",
        "what is today's date",
        "what day is it",
        "what day is today",
        "tell me the date",
        "what time is it",
        "what is the current date",
    ]
    for q in date_queries:
        res = await analyzer.analyze(q)
        assert res.web_required is False, f"Query '{q}' should NOT require web search"
        assert res.tool_calling_required is False, f"Query '{q}' should NOT require tool calling"
        assert res.task_type == TaskType.CONVERSATION
        assert res.complexity <= 0.2


@pytest.mark.asyncio
async def test_heuristic_analyzer_live_search_queries():
    analyzer = HeuristicFallbackAnalyzer()
    search_queries = [
        "What is the stock price of Apple right now?",
        "What's the weather in Tokyo?",
        "Who won the 2026 Super Bowl?",
        "Search google for quantum computing breakthroughs",
    ]
    for q in search_queries:
        res = await analyzer.analyze(q)
        assert res.web_required is True, f"Query '{q}' should require web search"
        assert res.tool_calling_required is True


@pytest.mark.asyncio
async def test_heuristic_analyzer_temporal_without_search():
    analyzer = HeuristicFallbackAnalyzer()
    # Mentioning 'today' or 'tomorrow' in creative/coding/general text should NOT trigger web search
    res1 = await analyzer.analyze("Today I want to write a story about an adventurous kitten")
    assert res1.web_required is False

    res2 = await analyzer.analyze("Can you write a python script for my project today?")
    assert res2.web_required is False
    assert res2.task_type == TaskType.CODING


@pytest.mark.asyncio
async def test_composite_analyzer_prefers_heuristic():
    mock_primary = MagicMock()
    mock_primary.analyze = AsyncMock()

    composite = CompositeRequestAnalyzer(primary_analyzer=mock_primary, prefer_heuristic=True)
    res = await composite.analyze("What's the date?")
    # Must NOT call primary analyzer for confident heuristic queries
    mock_primary.analyze.assert_not_called()
    assert res.web_required is False
    assert res.task_type == TaskType.CONVERSATION


@pytest.mark.asyncio
async def test_heuristic_analyzer_news_queries():
    analyzer = HeuristicFallbackAnalyzer()
    news_queries = [
        "whats the news in nigeria?",
        "what are the latest headlines?",
        "what is happening in Nigeria right now?",
        "tell me the news",
    ]
    for q in news_queries:
        res = await analyzer.analyze(q)
        assert res.web_required is True, f"Query '{q}' should require web search"
        assert res.tool_calling_required is True
