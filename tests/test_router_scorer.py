"""Unit tests for ScoringEngine component."""

from app.ai.router.policies import PolicyRegistry
from app.ai.router.schemas import RequestAnalysis, TaskType
from app.ai.router.scorer import ScoringEngine
from app.models.config import AppConfigDB, RoutingMode


def test_scorer_prefers_lightweight_model_for_simple_task():
    cfg = AppConfigDB()
    scorer = ScoringEngine()
    analysis = RequestAnalysis(
        task_type=TaskType.CONVERSATION,
        complexity=0.15,
        reasoning_required=0.1,
        coding_required=0.0,
    )
    policy = PolicyRegistry.get(RoutingMode.FAST)

    ranked = scorer.score_candidates(cfg.models_list, analysis, policy)
    assert len(ranked) > 0
    top_model = ranked[0]
    # Simple conversation with fast policy should pick flash-lite tier
    assert "flash-lite" in top_model.model_id
    assert top_model.breakdown.complexity_match >= 0.8


def test_scorer_prefers_powerful_model_for_complex_reasoning_task():
    cfg = AppConfigDB()
    scorer = ScoringEngine()
    analysis = RequestAnalysis(
        task_type=TaskType.CODING,
        complexity=0.92,
        reasoning_required=0.95,
        coding_required=0.95,
    )
    policy = PolicyRegistry.get(RoutingMode.QUALITY)

    ranked = scorer.score_candidates(cfg.models_list, analysis, policy)
    assert len(ranked) > 0
    top_model = ranked[0]
    # Deep coding & reasoning with quality policy should choose pro or 3.6-flash
    assert top_model.model_id in ("gemini-3.1-pro-preview", "gemini-3.6-flash")


def test_scorer_penalizes_under_capable_models():
    cfg = AppConfigDB()
    scorer = ScoringEngine()
    analysis = RequestAnalysis(
        task_type=TaskType.ANALYSIS,
        complexity=0.95,
        reasoning_required=0.90,
    )
    policy = PolicyRegistry.get(RoutingMode.BALANCED)

    ranked = scorer.score_candidates(cfg.models_list, analysis, policy)
    scored_dict = {c.model_id: c for c in ranked}

    # gemini-3.1-flash-lite (complexity=0.25) should have complexity_match ~ 0.0
    assert scored_dict["gemini-3.1-flash-lite"].breakdown.complexity_match == 0.0
    # gemini-3.1-pro-preview (complexity=0.95) should have high complexity_match
    assert scored_dict["gemini-3.1-pro-preview"].breakdown.complexity_match >= 0.90


def test_scorer_explainable_breakdown():
    cfg = AppConfigDB()
    scorer = ScoringEngine()
    analysis = RequestAnalysis(task_type=TaskType.GENERAL_KNOWLEDGE, complexity=0.5)
    policy = PolicyRegistry.get(RoutingMode.BALANCED)

    ranked = scorer.score_candidates(cfg.models_list, analysis, policy)
    for candidate in ranked:
        bd = candidate.breakdown
        assert 0.0 <= bd.capability_match <= 1.0
        assert 0.0 <= bd.complexity_match <= 1.0
        assert 0.0 <= bd.cost_score <= 1.0
        assert 0.0 <= bd.speed_score <= 1.0
        assert "capability" in bd.weighted_scores
        assert "cost" in bd.details
