"""Unit tests for CandidateFilter component."""

from app.ai.router.filters import CandidateFilter
from app.ai.router.schemas import RequestAnalysis, RequestContext, RoutingPolicy
from app.models.config import AppConfigDB, ModelStatus, ReasoningLevel, TextModelConfig


def test_filter_disabled_and_inactive_models():
    models = {
        "active-model": TextModelConfig(
            description="", enabled=True, reasoning_modes=[ReasoningLevel.NONE], status=ModelStatus.ACTIVE
        ),
        "disabled-model": TextModelConfig(
            description="", enabled=False, reasoning_modes=[ReasoningLevel.NONE], status=ModelStatus.ACTIVE
        ),
        "deprecated-model": TextModelConfig(
            description="", enabled=True, reasoning_modes=[ReasoningLevel.NONE], status=ModelStatus.DEPRECATED
        ),
    }
    filter_engine = CandidateFilter()
    analysis = RequestAnalysis()
    policy = RoutingPolicy()

    eligible, filtered = filter_engine.filter_candidates(models, analysis, policy)
    assert "active-model" in eligible
    assert "disabled-model" not in eligible
    assert "deprecated-model" not in eligible
    assert "disabled" in filtered["disabled-model"].lower()
    assert "deprecated" in filtered["deprecated-model"].lower()


def test_filter_allowed_models_policy():
    cfg = AppConfigDB()
    filter_engine = CandidateFilter()
    analysis = RequestAnalysis()
    policy = RoutingPolicy(allowed_models=["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"])

    eligible, filtered = filter_engine.filter_candidates(cfg.models_list, analysis, policy)
    assert set(eligible.keys()) == {"gemini-3.5-flash-lite", "gemini-3.1-flash-lite"}
    assert "gemini-3.6-flash" in filtered


def test_filter_vision_requirement():
    models = {
        "vision-model": TextModelConfig(description="", enabled=True, reasoning_modes=["none"], supports_vision=True),
        "text-only-model": TextModelConfig(
            description="", enabled=True, reasoning_modes=["none"], supports_vision=False
        ),
    }
    filter_engine = CandidateFilter()
    analysis = RequestAnalysis(vision_required=True)
    policy = RoutingPolicy()

    eligible, filtered = filter_engine.filter_candidates(models, analysis, policy)
    assert "vision-model" in eligible
    assert "text-only-model" not in eligible
    assert "vision" in filtered["text-only-model"].lower()


def test_filter_tools_requirement():
    models = {
        "tool-model": TextModelConfig(description="", enabled=True, reasoning_modes=["none"], supports_tools=True),
        "no-tool-model": TextModelConfig(description="", enabled=True, reasoning_modes=["none"], supports_tools=False),
    }
    filter_engine = CandidateFilter()
    analysis = RequestAnalysis(tool_calling_required=True)
    policy = RoutingPolicy()

    eligible, _filtered = filter_engine.filter_candidates(models, analysis, policy)
    assert "tool-model" in eligible
    assert "no-tool-model" not in eligible


def test_filter_context_window_limit():
    models = {
        "big-window": TextModelConfig(
            description="", enabled=True, reasoning_modes=["none"], context_window_tokens=1_000_000
        ),
        "small-window": TextModelConfig(
            description="", enabled=True, reasoning_modes=["none"], context_window_tokens=32_000
        ),
    }
    filter_engine = CandidateFilter()
    analysis = RequestAnalysis()
    policy = RoutingPolicy()
    ctx = RequestContext(approximate_context_tokens=100_000)

    eligible, _filtered = filter_engine.filter_candidates(models, analysis, policy, context=ctx)
    assert "big-window" in eligible
    assert "small-window" not in eligible


def test_filter_max_cost_limit():
    models = {
        "cheap-model": TextModelConfig(
            description="", enabled=True, reasoning_modes=["none"], input_cost_per_million=0.05
        ),
        "expensive-model": TextModelConfig(
            description="", enabled=True, reasoning_modes=["none"], input_cost_per_million=2.50
        ),
    }
    filter_engine = CandidateFilter()
    analysis = RequestAnalysis()
    policy = RoutingPolicy(max_cost=0.50)

    eligible, _filtered = filter_engine.filter_candidates(models, analysis, policy)
    assert "cheap-model" in eligible
    assert "expensive-model" not in eligible
