"""Candidate filtering components for pre-scoring model eligibility."""

from __future__ import annotations

import logging
from collections.abc import Mapping

from app.ai.router.schemas import RequestAnalysis, RequestContext, RoutingPolicy
from app.models.config import ModelStatus, TextModelConfig

logger = logging.getLogger(__name__)


class CandidateFilter:
    """Filters candidate models against hard constraints and required capabilities."""

    def filter_candidates(
        self,
        models: Mapping[str, TextModelConfig],
        analysis: RequestAnalysis,
        policy: RoutingPolicy,
        context: RequestContext | None = None,
        blacklisted_models: set[str] | None = None,
    ) -> tuple[dict[str, TextModelConfig], dict[str, str]]:
        """Filter out models that cannot satisfy the request.

        Returns:
            A tuple of (eligible_models, filtered_reasons) where:
            - eligible_models: dict of model_id -> TextModelConfig that passed all checks
            - filtered_reasons: dict of model_id -> reason string explaining why it was excluded
        """
        eligible: dict[str, TextModelConfig] = {}
        filtered: dict[str, str] = {}

        approx_context_tokens = context.approximate_context_tokens if context else 0

        for model_id, model in models.items():
            # 0. Routing pseudo-models
            if model_id.lower() in ("auto", "smart", "default", "none"):
                filtered[model_id] = "Auto is a routing selector, not an underlying physical model."
                continue

            # 1. Operational status and enabled toggle
            if not model.enabled:
                filtered[model_id] = "Model is disabled in system configuration."
                continue

            if model.status != ModelStatus.ACTIVE:
                filtered[model_id] = f"Model status is '{model.status}' (must be '{ModelStatus.ACTIVE.value}')."
                continue

            if blacklisted_models and model_id in blacklisted_models:
                filtered[model_id] = "Model is currently blacklisted due to recent spike errors or failures."
                continue

            # 2. User plan / allowed models constraint
            if policy.allowed_models is not None and model_id not in policy.allowed_models:
                filtered[model_id] = "Model is not allowed under the user's plan or policy restrictions."
                continue

            # 3. Provider restrictions
            if policy.preferred_providers is not None and model.provider not in policy.preferred_providers:
                filtered[model_id] = (
                    f"Model provider '{model.provider}' is not in allowed providers {policy.preferred_providers}."
                )
                continue

            # 4. Vision capability
            if analysis.vision_required and not model.supports_vision:
                filtered[model_id] = "Request requires vision capabilities, which this model does not support."
                continue

            # 5. Tool / Function calling capability
            if (analysis.tool_calling_required or analysis.web_required) and not model.supports_tools:
                filtered[model_id] = "Request requires tool/grounding support, which this model does not support."
                continue

            # 6. Structured output capability
            if analysis.structured_output_required and not model.supports_structured_output:
                filtered[model_id] = "Request requires structured JSON output, which this model does not support."
                continue

            # 7. Context window limit
            if approx_context_tokens > 0 and approx_context_tokens > model.context_window_tokens:
                filtered[model_id] = (
                    f"Context length ({approx_context_tokens} tokens) exceeds model limit "
                    f"({model.context_window_tokens} tokens)."
                )
                continue

            # 8. Maximum cost constraint
            if policy.max_cost is not None and model.input_cost_per_million > policy.max_cost:
                filtered[model_id] = (
                    f"Model cost (${model.input_cost_per_million}/1M tokens) exceeds max allowed cost (${policy.max_cost})."
                )
                continue

            # Passed all hard constraint checks
            eligible[model_id] = model

        logger.debug(
            "Candidate filtering completed: %d eligible, %d excluded",
            len(eligible),
            len(filtered),
        )
        return eligible, filtered
