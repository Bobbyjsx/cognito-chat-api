"""Unit tests for Smart Model Router data schemas and models."""

from app.ai.router.schemas import (
    PolicyWeights,
    RequestAnalysis,
    RequestContext,
    RoutingDecision,
    RoutingTelemetry,
    ScoreBreakdown,
    ScoredCandidate,
    TaskType,
)
from app.models.attachments import AttachmentType
from app.models.config import ModelProvider, ReasoningLevel, RoutingMode, TextModelConfig


def test_request_analysis_defaults():
    analysis = RequestAnalysis()
    assert analysis.task_type == TaskType.GENERAL_KNOWLEDGE
    assert 0.0 <= analysis.complexity <= 1.0
    assert analysis.vision_required is False
    assert analysis.tool_calling_required is False
    assert analysis.confidence == 1.0


def test_request_context_creation():
    ctx = RequestContext(
        conversation_message_count=5,
        approximate_context_tokens=1200,
        has_attachments=True,
        attachment_types=[AttachmentType.image],
        user_id="user-123",
        session_id="session-456",
    )
    assert ctx.conversation_message_count == 5
    assert ctx.has_attachments is True
    assert AttachmentType.image in ctx.attachment_types


def test_policy_weights_validation():
    weights = PolicyWeights(
        capability_weight=0.30,
        complexity_weight=0.30,
        quality_weight=0.20,
        speed_weight=0.10,
        cost_weight=0.10,
    )
    assert weights.capability_weight == 0.30
    assert weights.speed_weight == 0.10


def test_score_breakdown_and_candidate():
    breakdown = ScoreBreakdown(
        capability_match=0.9,
        complexity_match=0.95,
        quality_score=0.88,
        speed_score=0.85,
        cost_score=0.80,
        reliability_score=1.0,
        raw_scores={"speed": 0.85},
        weighted_scores={"speed": 0.1275},
        details={"reason": "test"},
    )
    model_cfg = TextModelConfig(
        description="Test model",
        enabled=True,
        reasoning_modes=[ReasoningLevel.FAST, ReasoningLevel.BALANCED],
    )
    candidate = ScoredCandidate(
        model_id="test-model",
        provider=ModelProvider.GOOGLE,
        total_score=0.89,
        rank=1,
        breakdown=breakdown,
        model_config=model_cfg,
    )
    assert candidate.model_id == "test-model"
    assert candidate.rank == 1
    assert candidate.breakdown.capability_match == 0.9
    assert candidate.provider == ModelProvider.GOOGLE


def test_routing_decision_and_telemetry():
    analysis = RequestAnalysis(task_type=TaskType.CODING, complexity=0.8)
    breakdown = ScoreBreakdown(
        capability_match=0.9,
        complexity_match=0.85,
        quality_score=0.9,
        speed_score=0.7,
        cost_score=0.6,
        reliability_score=1.0,
    )
    decision = RoutingDecision(
        selected_model_id="gemini-3.6-flash",
        provider=ModelProvider.GOOGLE,
        score=0.86,
        routing_mode=RoutingMode.BALANCED,
        task_type=TaskType.CODING,
        complexity=0.8,
        reason=breakdown,
        analysis=analysis,
        fallback_models=["gemini-3.5-flash"],
    )
    assert decision.selected_model_id == "gemini-3.6-flash"
    assert decision.provider == ModelProvider.GOOGLE
    assert decision.routing_mode == RoutingMode.BALANCED
    assert decision.task_type == TaskType.CODING
    assert decision.fallback_models == ["gemini-3.5-flash"]
    assert decision.is_fallback_decision is False

    telemetry = RoutingTelemetry(
        selected_model=decision.selected_model_id,
        provider=decision.provider,
        routing_mode=decision.routing_mode,
        task_type=decision.task_type,
        complexity=decision.complexity,
        score=decision.score,
        candidate_models=["gemini-3.6-flash", "gemini-3.5-flash"],
    )
    assert telemetry.request_id.startswith("route_")
    assert telemetry.selected_model == "gemini-3.6-flash"
    assert telemetry.provider == ModelProvider.GOOGLE
    assert telemetry.routing_mode == RoutingMode.BALANCED
    assert telemetry.task_type == TaskType.CODING
