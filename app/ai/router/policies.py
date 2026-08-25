"""Routing policies and preset definitions for the Smart Model Router."""

from __future__ import annotations

import logging
from typing import ClassVar

from app.ai.router.schemas import PolicyWeights, RoutingPolicy
from app.models.config import RoutingMode

logger = logging.getLogger(__name__)


class PolicyRegistry:
    """Registry of standard routing policies with tuned weights."""

    PRESETS: ClassVar[dict[RoutingMode, RoutingPolicy]] = {
        RoutingMode.FAST: RoutingPolicy(
            mode=RoutingMode.FAST,
            weights=PolicyWeights(
                speed_weight=0.40,
                cost_weight=0.30,
                capability_weight=0.15,
                complexity_weight=0.10,
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
        RoutingMode.EXTENDED: RoutingPolicy(
            mode=RoutingMode.EXTENDED,
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

    # Aliases mapping legacy names to unified modes
    _ALIASES: ClassVar[dict[str, RoutingMode]] = {
        "speed": RoutingMode.FAST,
        "cost": RoutingMode.FAST,
        "fast": RoutingMode.FAST,
        "balanced": RoutingMode.BALANCED,
        "medium": RoutingMode.BALANCED,
        "quality": RoutingMode.EXTENDED,
        "extended": RoutingMode.EXTENDED,
        "high": RoutingMode.EXTENDED,
        "deep": RoutingMode.EXTENDED,
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

        # Check aliases first
        if mode_val in cls._ALIASES:
            enum_mode = cls._ALIASES[mode_val]
            return cls.PRESETS[enum_mode].model_copy(deep=True)

        try:
            enum_mode = RoutingMode(mode_val)
            if enum_mode in cls.PRESETS:
                return cls.PRESETS[enum_mode].model_copy(deep=True)
        except ValueError:
            logger.warning("Unrecognized routing policy mode '%s'. Defaulting to '%s'.", mode_val, default_val)

        # Fallback to default
        fallback_mode = cls._ALIASES.get(default_val, RoutingMode.BALANCED)
        return cls.PRESETS[fallback_mode].model_copy(deep=True)
