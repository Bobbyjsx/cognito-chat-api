"""Deterministic Scoring Engine for ranking candidate AI models."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from app.ai.router.schemas import (
    CapacityStatus,
    RequestAnalysis,
    RoutingPolicy,
    ScoreBreakdown,
    ScoredCandidate,
    ScoreDimension,
)
from app.models.config import ModelStatus, TextModelConfig

logger = logging.getLogger(__name__)


class ScoringEngine:
    """Evaluates and ranks candidate models against request requirements and routing policies."""

    def score_candidates(
        self,
        candidate_models: Mapping[str, TextModelConfig],
        analysis: RequestAnalysis,
        policy: RoutingPolicy,
    ) -> list[ScoredCandidate]:
        """Score all eligible candidate models and return them sorted by score descending."""
        if not candidate_models:
            return []

        # Precompute cost scale across the candidate pool for relative normalization
        min_cost, max_cost = self._compute_cost_range(candidate_models)

        scored: list[ScoredCandidate] = []
        for model_id, model in candidate_models.items():
            breakdown = self.evaluate_model(
                model=model,
                analysis=analysis,
                policy=policy,
                min_pool_cost=min_cost,
                max_pool_cost=max_cost,
            )

            # Compute weighted sum
            weights = policy.weights
            total_score = (
                breakdown.capability_match * weights.capability_weight
                + breakdown.complexity_match * weights.complexity_weight
                + breakdown.quality_score * weights.quality_weight
                + breakdown.speed_score * weights.speed_weight
                + breakdown.cost_score * weights.cost_weight
                + breakdown.reliability_score * weights.reliability_weight
            )

            # Update weighted scores map for full explainability
            breakdown.weighted_scores = {
                ScoreDimension.CAPABILITY.value: round(breakdown.capability_match * weights.capability_weight, 4),
                ScoreDimension.COMPLEXITY.value: round(breakdown.complexity_match * weights.complexity_weight, 4),
                ScoreDimension.QUALITY.value: round(breakdown.quality_score * weights.quality_weight, 4),
                ScoreDimension.SPEED.value: round(breakdown.speed_score * weights.speed_weight, 4),
                ScoreDimension.COST.value: round(breakdown.cost_score * weights.cost_weight, 4),
                ScoreDimension.RELIABILITY.value: round(breakdown.reliability_score * weights.reliability_weight, 4),
            }

            candidate = ScoredCandidate(
                model_id=model_id,
                provider=model.provider,
                total_score=round(total_score, 4),
                rank=0,  # assigned after sorting
                breakdown=breakdown,
                model_config=model,
            )
            scored.append(candidate)

        # Sort descending by total score, with speed and cost as deterministic tiebreakers
        scored.sort(
            key=lambda c: (
                c.total_score,
                c.breakdown.quality_score,
                c.breakdown.speed_score,
                c.breakdown.cost_score,
            ),
            reverse=True,
        )

        # Assign 1-indexed ranks
        for rank, candidate in enumerate(scored, start=1):
            candidate.rank = rank

        return scored

    def evaluate_model(
        self,
        model: TextModelConfig,
        analysis: RequestAnalysis,
        policy: RoutingPolicy,
        min_pool_cost: float,
        max_pool_cost: float,
    ) -> ScoreBreakdown:
        """Compute the detailed multidimensional score breakdown for a single model."""
        cap_match, cap_details = self._calculate_capability_match(model, analysis)
        comp_match, comp_details = self._calculate_complexity_match(model, analysis)
        cost_sc, cost_details = self._calculate_cost_score(model, min_pool_cost, max_pool_cost)
        speed_sc = model.speed_score
        quality_sc = model.quality_score
        reliability_sc = 1.0 if model.status == ModelStatus.ACTIVE else 0.5

        raw_scores = {
            "capability_match": round(cap_match, 4),
            "complexity_match": round(comp_match, 4),
            "quality_score": round(quality_sc, 4),
            "speed_score": round(speed_sc, 4),
            "cost_score": round(cost_sc, 4),
            "reliability_score": round(reliability_sc, 4),
        }

        details = {
            "capabilities": cap_details,
            "complexity": comp_details,
            "cost": cost_details,
            "pricing": {
                "input_cost_per_million": model.input_cost_per_million,
                "output_cost_per_million": model.output_cost_per_million,
            },
        }

        return ScoreBreakdown(
            capability_match=round(cap_match, 4),
            complexity_match=round(comp_match, 4),
            quality_score=round(quality_sc, 4),
            speed_score=round(speed_sc, 4),
            cost_score=round(cost_sc, 4),
            reliability_score=round(reliability_sc, 4),
            raw_scores=raw_scores,
            weighted_scores={},
            details=details,
        )

    def _calculate_capability_match(
        self,
        model: TextModelConfig,
        analysis: RequestAnalysis,
    ) -> tuple[float, dict[str, float]]:
        """Calculate how well the model's specialized capabilities align with the request."""
        scores: list[float] = []
        weights: list[float] = []
        details: dict[str, float] = {}

        # 1. Coding alignment
        if analysis.coding_required > 0:
            deficit = max(0.0, analysis.coding_required - model.coding_score)
            match_score = max(0.0, 1.0 - (deficit * 2.0))
            scores.append(match_score)
            weights.append(analysis.coding_required * 2.0)
            details["coding_match"] = round(match_score, 3)

        # 2. Reasoning alignment
        if analysis.reasoning_required > 0:
            deficit = max(0.0, analysis.reasoning_required - model.reasoning_score)
            match_score = max(0.0, 1.0 - (deficit * 2.0))
            scores.append(match_score)
            weights.append(analysis.reasoning_required * 1.5)
            details["reasoning_match"] = round(match_score, 3)

        # 3. Creative alignment
        if analysis.creative_required > 0:
            deficit = max(0.0, analysis.creative_required - model.creative_score)
            match_score = max(0.0, 1.0 - (deficit * 1.5))
            scores.append(match_score)
            weights.append(analysis.creative_required * 1.0)
            details["creative_match"] = round(match_score, 3)

        # 4. Context alignment
        if analysis.context_required > 0:
            deficit = max(0.0, analysis.context_required - model.context_score)
            match_score = max(0.0, 1.0 - (deficit * 1.5))
            scores.append(match_score)
            weights.append(analysis.context_required * 1.0)
            details["context_match"] = round(match_score, 3)

        # 5. Vision score
        if analysis.vision_required:
            scores.append(model.vision_score)
            weights.append(1.5)
            details["vision_score"] = round(model.vision_score, 3)

        # 6. Tool calling score
        if analysis.tool_calling_required or analysis.web_required:
            scores.append(model.tool_calling_score)
            weights.append(1.0)
            details["tool_calling_score"] = round(model.tool_calling_score, 3)

        # Base quality component ensures general requests have strong baseline match
        base_weight = 1.0
        scores.append(model.quality_score)
        weights.append(base_weight)
        details["base_quality"] = round(model.quality_score, 3)

        total_weight = sum(weights)
        if total_weight == 0:
            return model.quality_score, details

        final_match = sum(s * w for s, w in zip(scores, weights)) / total_weight
        return min(1.0, max(0.0, final_match)), details

    def _calculate_complexity_match(
        self,
        model: TextModelConfig,
        analysis: RequestAnalysis,
    ) -> tuple[float, dict[str, Any]]:
        """Calculate complexity match.

        Principles:
        - We want the lowest-cost model that COMFORTABLY satisfies the request.
        - If model complexity >= request complexity:
            Capable model. Mild over-provisioning discount to naturally favor
            right-sized models over bloated ones.
        - If model complexity < request complexity:
            Under-capable model. Steep penalty so simple models are not picked
            for high-complexity reasoning tasks.
        """
        req_comp = analysis.complexity
        model_comp = model.complexity_score
        delta = model_comp - req_comp

        if delta >= 0:
            # Model comfortably handles the task.
            # Right-sizing discount: gently favor right-sized models over bloated ones for simple tasks
            score = max(0.50, 1.0 - (0.60 * delta))
            status = CapacityStatus.SUFFICIENT_CAPACITY.value
        else:
            # Model has a capacity deficit
            # E.g. req=0.9, model=0.85 -> deficit=0.05 -> score=0.825
            #      req=0.9, model=0.70 -> deficit=0.20 -> score=0.300
            #      req=0.9, model=0.40 -> deficit=0.50 -> score=0.000
            deficit = abs(delta)
            score = max(0.0, 1.0 - (3.5 * deficit))
            status = CapacityStatus.CAPACITY_DEFICIT.value

        details = {
            "model_complexity": model_comp,
            "request_complexity": req_comp,
            "delta": round(delta, 3),
            "capacity_status": status,
        }
        return score, details

    def _compute_cost_range(
        self,
        models: Mapping[str, TextModelConfig],
    ) -> tuple[float, float]:
        """Compute the min and max effective price per 1M tokens across the candidate pool."""
        prices = [m.input_cost_per_million + (2.0 * m.output_cost_per_million) for m in models.values()]
        if not prices:
            return 0.0, 1.0
        return min(prices), max(prices)

    def _calculate_cost_score(
        self,
        model: TextModelConfig,
        min_cost: float,
        max_cost: float,
    ) -> tuple[float, dict[str, Any]]:
        """Normalize model price to a 0.0 - 1.0 score where higher is cheaper/better value."""
        model_effective = model.input_cost_per_million + (2.0 * model.output_cost_per_million)

        if max_cost <= min_cost or max_cost == 0:
            return 1.0, {"effective_cost": model_effective}

        # Scale linearly between 1.0 (cheapest) and 0.15 (most expensive in pool)
        normalized = 1.0 - (0.85 * ((model_effective - min_cost) / (max_cost - min_cost)))
        score = min(1.0, max(0.10, normalized))

        details = {
            "effective_cost_per_million": round(model_effective, 4),
            "min_pool_cost": round(min_cost, 4),
            "max_pool_cost": round(max_cost, 4),
        }
        return score, details
