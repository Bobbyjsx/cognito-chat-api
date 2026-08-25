"""Routing policies and preset definitions for the Smart Model Router."""

from __future__ import annotations

import logging
from typing import ClassVar

from app.ai.router.exceptions import InvalidPolicyError
from app.ai.router.schemas import PolicyWeights, RoutingPolicy
from app.models.config import RoutingMode

logger = logging.getLogger(__name__)


class PolicyRegistry:
    """Registry of standard routing policies with tuned weights."""

    PRESETS: ClassVar[dict[RoutingMode, RoutingPolicy]] = {
        RoutingMode.FAST: RoutingPolicy(
            mode=RoutingMode.FAST,
            weights=PolicyWeights(
                speed_weight=0.35,
                cost_weight=0.30,
                capability_weight=0.15,
                complexity_weight=0.15,
                quality_weight=0.05,
                reliability_weight=0.00,
            ),
        ),
        RoutingMode.BALANCED: RoutingPolicy(
            mode=RoutingMode.BALANCED,
            weights=PolicyWeights(
                capability_weight=0.25,
                complexity_weight=0.25,
                quality_weight=0.20,
                speed_weight=0.15,
                cost_weight=0.15,
                reliability_weight=0.00,
            ),
        ),
        RoutingMode.QUALITY: RoutingPolicy(
            mode=RoutingMode.QUALITY,
            weights=PolicyWeights(
                quality_weight=0.35,
                capability_weight=0.30,
                complexity_weight=0.25,
                speed_weight=0.05,
                cost_weight=0.05,
                reliability_weight=0.00,
            ),
        ),
    }

    @classmethod
    def get(
        cls,
        policy_or_mode: RoutingMode | str | RoutingPolicy | None,
        default_mode: RoutingMode | str = RoutingMode.BALANCED,
    ) -> RoutingPolicy:
        """Resolve a policy name or instance into a valid RoutingPolicy."""
        if isinstance(policy_or_mode, RoutingPolicy):
            return policy_or_mode

        target = policy_or_mode if policy_or_mode is not None else default_mode
        mode_val = (target.value if isinstance(target, RoutingMode) else str(target)).lower().strip()
        default_val = (
            (default_mode.value if isinstance(default_mode, RoutingMode) else str(default_mode)).lower().strip()
        )

        try:
            enum_mode = RoutingMode(mode_val)
            if enum_mode in cls.PRESETS:
                return cls.PRESETS[enum_mode].model_copy(deep=True)
        except ValueError:
            logger.warning("Unrecognized routing policy mode '%s'. Defaulting to '%s'.", mode_val, default_val)

        try:
            fallback_enum = RoutingMode(default_val)
            if fallback_enum in cls.PRESETS:
                return cls.PRESETS[fallback_enum].model_copy(deep=True)
        except ValueError:
            pass

        raise InvalidPolicyError(f"Invalid routing mode '{mode_val}' and default '{default_val}'")
