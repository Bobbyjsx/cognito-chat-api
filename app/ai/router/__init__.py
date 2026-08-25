"""Smart Model Router module for dynamic, capability-based model selection."""

from app.ai.router.analyzer import (
    BaseRequestAnalyzer,
    CompositeRequestAnalyzer,
    GeminiFlashLiteAnalyzer,
    HeuristicFallbackAnalyzer,
)
from app.ai.router.exceptions import (
    AnalyzerError,
    InvalidPolicyError,
    NoEligibleModelsError,
    RoutingError,
)
from app.ai.router.filters import CandidateFilter
from app.ai.router.policies import PolicyRegistry
from app.ai.router.router import SmartModelRouter
from app.ai.router.schemas import (
    CapacityStatus,
    PolicyWeights,
    RequestAnalysis,
    RequestContext,
    RoutingDecision,
    RoutingPolicy,
    RoutingTelemetry,
    ScoreBreakdown,
    ScoredCandidate,
    ScoreDimension,
    TaskType,
)
from app.ai.router.scorer import ScoringEngine
from app.ai.router.telemetry import (
    CompositeTelemetrySink,
    InMemoryTelemetrySink,
    LoggingTelemetrySink,
    TelemetrySink,
)
from app.models.config import ModelProvider, ModelStatus, ReasoningLevel, RoutingMode

__all__ = [
    "AnalyzerError",
    "BaseRequestAnalyzer",
    "CandidateFilter",
    "CapacityStatus",
    "CompositeRequestAnalyzer",
    "CompositeTelemetrySink",
    "GeminiFlashLiteAnalyzer",
    "HeuristicFallbackAnalyzer",
    "InMemoryTelemetrySink",
    "InvalidPolicyError",
    "LoggingTelemetrySink",
    "ModelProvider",
    "ModelStatus",
    "NoEligibleModelsError",
    "PolicyRegistry",
    "PolicyWeights",
    "ReasoningLevel",
    "RequestAnalysis",
    "RequestContext",
    "RoutingDecision",
    "RoutingError",
    "RoutingMode",
    "RoutingPolicy",
    "RoutingTelemetry",
    "ScoreBreakdown",
    "ScoreDimension",
    "ScoredCandidate",
    "ScoringEngine",
    "SmartModelRouter",
    "TaskType",
    "TelemetrySink",
]
