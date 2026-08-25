"""Unit tests for PolicyRegistry and routing policy resolution."""

from app.ai.router.policies import PolicyRegistry
from app.ai.router.schemas import PolicyWeights, RoutingPolicy
from app.models.config import RoutingMode


def test_policy_registry_presets():
    fast = PolicyRegistry.get(RoutingMode.FAST)
    assert fast.mode == RoutingMode.FAST
    assert fast.weights.speed_weight > fast.weights.quality_weight

    extended = PolicyRegistry.get(RoutingMode.EXTENDED)
    assert extended.mode == RoutingMode.EXTENDED
    assert extended.weights.quality_weight > extended.weights.speed_weight

    balanced = PolicyRegistry.get("balanced")
    assert balanced.mode == RoutingMode.BALANCED


def test_policy_registry_custom_instance():
    custom = RoutingPolicy(
        mode=RoutingMode.CUSTOM,
        weights=PolicyWeights(capability_weight=0.5, complexity_weight=0.5),
        max_cost=1.0,
    )
    res = PolicyRegistry.get(custom)
    assert res.mode == RoutingMode.CUSTOM
    assert res.max_cost == 1.0


def test_policy_registry_unknown_mode_fallback():
    fallback = PolicyRegistry.get("non-existent-mode", default_mode=RoutingMode.BALANCED)
    assert fallback.mode == RoutingMode.BALANCED
