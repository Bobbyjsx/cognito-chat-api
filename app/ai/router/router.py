"""Smart Model Router — Core Orchestration Engine."""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping

from app.ai.router.analyzer import (
    BaseRequestAnalyzer,
    CompositeRequestAnalyzer,
    GeminiFlashLiteAnalyzer,
    HeuristicFallbackAnalyzer,
)
from app.ai.router.exceptions import NoEligibleModelsError, RoutingError
from app.ai.router.filters import CandidateFilter
from app.ai.router.policies import PolicyRegistry
from app.ai.router.schemas import (
    RequestContext,
    RoutingDecision,
    RoutingPolicy,
    RoutingTelemetry,
)
from app.ai.router.scorer import ScoringEngine
from app.ai.router.telemetry import (
    CompositeTelemetrySink,
    InMemoryTelemetrySink,
    LoggingTelemetrySink,
    TelemetrySink,
)
from app.models.config import AppConfigDB, TextModelConfig

logger = logging.getLogger(__name__)


class SmartModelRouter:
    """Intelligent Model Router that deterministically selects the optimal AI model

    combining lightweight request classification with candidate capability filtering,
    complexity matching, cost/speed trade-offs, and explainable telemetry.
    """

    def __init__(
        self,
        analyzer: BaseRequestAnalyzer | None = None,
        candidate_filter: CandidateFilter | None = None,
        scorer: ScoringEngine | None = None,
        telemetry_sink: TelemetrySink | None = None,
        default_policy_mode: str = "balanced",
    ):
        if analyzer is None:
            # Default composite analyzer with Flash-Lite primary and heuristic fallback
            flash_lite = GeminiFlashLiteAnalyzer()
            heuristic = HeuristicFallbackAnalyzer()
            analyzer = CompositeRequestAnalyzer(primary_analyzer=flash_lite, fallback_analyzer=heuristic)
        self.analyzer = analyzer

        self.filter = candidate_filter or CandidateFilter()
        self.scorer = scorer or ScoringEngine()

        if telemetry_sink is None:
            telemetry_sink = CompositeTelemetrySink([LoggingTelemetrySink(), InMemoryTelemetrySink()])
        self.telemetry_sink = telemetry_sink

        self.default_policy_mode = default_policy_mode

    async def route(
        self,
        message: str,
        context: RequestContext | None = None,
        policy: RoutingPolicy | str | None = None,
        config: AppConfigDB | None = None,
        models_override: Mapping[str, TextModelConfig] | None = None,
    ) -> RoutingDecision:
        """Route a user request to the best model using deterministic scoring."""
        start_time = time.perf_counter()
        analysis_latency_ms: float | None = None
        scoring_latency_ms: float | None = None

        # 1. Resolve configuration and active models
        if config is None:
            config = AppConfigDB()

        models: Mapping[str, TextModelConfig] = models_override if models_override is not None else config.models_list

        # 2. Resolve policy
        default_mode = config.default_routing_mode or self.default_policy_mode
        resolved_policy = PolicyRegistry.get(policy, default_mode=default_mode)

        logger.info(
            "[SmartRouter] Routing request: '%s...' (policy=%s)",
            message[:80].replace("\n", " ").strip(),
            resolved_policy.mode.value,
        )

        # 3. Analyze request
        analysis_start = time.perf_counter()
        try:
            analysis = await self.analyzer.analyze(message, context)
        except Exception as exc:
            err_type = type(exc).__name__
            err_msg = getattr(exc, "message", None) or str(exc) or repr(exc)
            logger.warning("[SmartRouter] Request analysis failed (%s: %s); using baseline heuristic.", err_type, err_msg)
            fallback = HeuristicFallbackAnalyzer()
            analysis = await fallback.analyze(message, context)
        analysis_latency_ms = (time.perf_counter() - analysis_start) * 1000.0

        # 4. Filter candidate models
        eligible_models, filtered_reasons = self.filter.filter_candidates(
            models=models,
            analysis=analysis,
            policy=resolved_policy,
            context=context,
        )
        logger.info(
            "[SmartRouter][Filter] %d/%d candidate models eligible. Excluded: %s",
            len(eligible_models),
            len(models),
            filtered_reasons if filtered_reasons else "none",
        )

        # 5. Handle no eligible models edge case
        if not eligible_models:
            error_msg = (
                f"No eligible models found for request (task={analysis.task_type.value}, complexity={analysis.complexity}). "
                f"Filtered {len(filtered_reasons)} candidates: {filtered_reasons}"
            )
            logger.error(error_msg)

            # Record telemetry for failed routing
            total_latency_ms = (time.perf_counter() - start_time) * 1000.0
            from app.models.config import ModelProvider

            telemetry = RoutingTelemetry(
                selected_model="none",
                provider=ModelProvider.OTHER,
                routing_mode=resolved_policy.mode,
                task_type=analysis.task_type,
                complexity=analysis.complexity,
                score=0.0,
                candidate_models=[],
                filtered_out_models=filtered_reasons,
                analysis_latency_ms=round(analysis_latency_ms, 2),
                scoring_latency_ms=0.0,
                total_latency_ms=round(total_latency_ms, 2),
                is_fallback=False,
            )
            await self.telemetry_sink.emit(telemetry)

            raise NoEligibleModelsError(error_msg, filtered_reasons=filtered_reasons)

        # 6. Score eligible candidates
        scoring_start = time.perf_counter()
        ranked_candidates = self.scorer.score_candidates(
            candidate_models=eligible_models,
            analysis=analysis,
            policy=resolved_policy,
        )
        scoring_latency_ms = (time.perf_counter() - scoring_start) * 1000.0

        if not ranked_candidates:
            raise RoutingError("Scoring engine produced no ranked candidates from eligible pool.")

        logger.info(
            "[SmartRouter][Scoring] Ranked candidates: %s",
            ", ".join(f"{c.model_id} (score={c.total_score:.3f}, rank={c.rank})" for c in ranked_candidates),
        )

        # 7. Select top candidate and extract fallback candidates
        top_candidate = ranked_candidates[0]
        fallback_models = [c.model_id for c in ranked_candidates[1:]]

        total_latency_ms = (time.perf_counter() - start_time) * 1000.0

        # 8. Produce decision and telemetry
        decision = RoutingDecision(
            selected_model_id=top_candidate.model_id,
            provider=top_candidate.provider,
            score=top_candidate.total_score,
            routing_mode=resolved_policy.mode,
            task_type=analysis.task_type,
            complexity=analysis.complexity,
            reason=top_candidate.breakdown,
            ranked_candidates=ranked_candidates,
            analysis=analysis,
            fallback_models=fallback_models,
            is_fallback_decision=False,
            metadata={
                "analysis_latency_ms": round(analysis_latency_ms, 2),
                "scoring_latency_ms": round(scoring_latency_ms, 2),
                "total_latency_ms": round(total_latency_ms, 2),
                "eligible_count": len(eligible_models),
                "filtered_count": len(filtered_reasons),
            },
        )

        logger.info(
            "[SmartRouter][Decision] Selected '%s' (provider=%s, score=%.4f, mode=%s, task=%s, complexity=%.2f) in %.1f ms | Breakdown: capability=%.2f, complexity=%.2f, quality=%.2f, speed=%.2f, cost=%.2f",
            decision.selected_model_id,
            decision.provider.value,
            decision.score,
            decision.routing_mode.value,
            decision.task_type.value,
            decision.complexity,
            total_latency_ms,
            decision.reason.capability_match,
            decision.reason.complexity_match,
            decision.reason.quality_score,
            decision.reason.speed_score,
            decision.reason.cost_score,
        )

        telemetry = RoutingTelemetry(
            selected_model=decision.selected_model_id,
            provider=decision.provider,
            routing_mode=decision.routing_mode,
            task_type=decision.task_type,
            complexity=decision.complexity,
            score=decision.score,
            candidate_models=[c.model_id for c in ranked_candidates],
            filtered_out_models=filtered_reasons,
            analysis_latency_ms=round(analysis_latency_ms, 2),
            scoring_latency_ms=round(scoring_latency_ms, 2),
            total_latency_ms=round(total_latency_ms, 2),
            is_fallback=False,
        )
        await self.telemetry_sink.emit(telemetry)

        return decision

    async def route_or_default(
        self,
        message: str,
        context: RequestContext | None = None,
        requested_model: str | None = None,
        policy: RoutingPolicy | str | None = None,
        config: AppConfigDB | None = None,
    ) -> tuple[str, list[str], RoutingDecision | None]:
        """Convenience method for chat services.

        If `requested_model` is explicitly specified by user (and not "auto"),
        it validates and returns that model.
        If `requested_model` is None, "auto", or smart routing is invoked,
        it performs intelligent routing with graceful fallback to the default model.

        Returns:
            A tuple of (selected_model_id, fallback_model_ids, routing_decision_or_none)
        """
        if config is None:
            config = AppConfigDB()

        is_explicit = requested_model and requested_model not in ("auto", "smart", "default")
        if is_explicit:
            logger.info("[SmartRouter] Explicit model '%s' requested by user. Bypassing smart routing.", requested_model)
            fallbacks = [m for m in config.allowed_text_models if m != requested_model]
            return requested_model, fallbacks, None

        # Execute smart routing
        try:
            decision = await self.route(
                message=message,
                context=context,
                policy=policy,
                config=config,
            )
            return decision.selected_model_id, decision.fallback_models, decision
        except Exception as exc:
            err_type = type(exc).__name__
            err_msg = getattr(exc, "message", None) or str(exc) or repr(exc)
            logger.warning(
                "[SmartRouter] Smart routing encountered an exception (%s: %s). Defaulting to system default '%s'.",
                err_type,
                err_msg,
                config.default_text_model,
            )
            fallbacks = [m for m in config.allowed_text_models if m != config.default_text_model]
            return config.default_text_model, fallbacks, None
