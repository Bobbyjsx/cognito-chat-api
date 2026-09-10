from datetime import datetime, timezone

from app.billing.models import SubscriptionDB, SubscriptionStatus

# Plan hierarchy: higher numbers include lower numbers
PLAN_HIERARCHY = {"free": 0, "go": 1, "premium": 2}


def get_entitlement_level(subscription: SubscriptionDB | None) -> int:
    if not subscription:
        return PLAN_HIERARCHY["free"]

    if subscription.status == SubscriptionStatus.ACTIVE:
        return PLAN_HIERARCHY.get(subscription.tier, 0)

    if subscription.status == SubscriptionStatus.CANCELLED and subscription.current_period_end:
        # Check if current_period_end is aware, if not make it aware
        end_time = subscription.current_period_end
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=timezone.utc)
        if end_time > datetime.now(timezone.utc):
            return PLAN_HIERARCHY.get(subscription.tier, 0)

    return PLAN_HIERARCHY["free"]


def has_minimum_tier(subscription: SubscriptionDB | None, required_tier: str) -> bool:
    required_level = PLAN_HIERARCHY.get(required_tier, 0)
    user_level = get_entitlement_level(subscription)
    return user_level >= required_level
