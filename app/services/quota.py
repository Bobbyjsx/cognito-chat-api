"""Quota resolution for Cognito.

Priority order for token limits:
  1. Per-user override (token_limit_6h / token_limit_weekly on UserDB)
  2. Canonical subscription-tier limits (never config — those can go stale)

Tier limits:
  Free     → 6h: 10,000   weekly: 100,000
  Go       → 6h: 40,000   weekly: 400,000
  Premium  → 6h: 60,000   weekly: 600,000
"""

from datetime import datetime, timezone

from app.models.config import AppConfigDB
from app.models.users import UserDB, UserResponse
from app.utils.datetime import ensure_utc

# Canonical tier limits — must match 001_set_tier_quotas migration.
TIER_LIMITS: dict[str, dict[str, int]] = {
    "free": {
        "token_limit_6h": 10_000,
        "token_limit_weekly": 100_000,
    },
    "go": {
        "token_limit_6h": 40_000,
        "token_limit_weekly": 400_000,
    },
    "premium": {
        "token_limit_6h": 60_000,
        "token_limit_weekly": 600_000,
    },
}


def normalize_tier(tier: str | None) -> str:
    if not tier:
        return "free"
    key = tier.lower().strip()
    return key if key in TIER_LIMITS else "free"


def _tier_limits(tier: str | None) -> tuple[int, int]:
    limits = TIER_LIMITS[normalize_tier(tier)]
    return limits["token_limit_6h"], limits["token_limit_weekly"]


def resolve_user_limits(
    user: UserDB,
    config: AppConfigDB | None = None,
    subscription_tier: str | None = None,
) -> tuple[int, int]:
    """Return (limit_6h, limit_weekly) for the given user.

    `config` is accepted for call-site compatibility but is not used for
    enforcement — stale app_config defaults must never inflate a free user
    to paid limits.
    """
    del config  # canonical TIER_LIMITS only
    tier_6h, tier_weekly = _tier_limits(subscription_tier)
    limit_6h = user.token_limit_6h if user.token_limit_6h is not None else tier_6h
    limit_weekly = user.token_limit_weekly if user.token_limit_weekly is not None else tier_weekly
    return limit_6h, limit_weekly


def format_countdown_string(dt_val: datetime | None) -> str:
    if not dt_val:
        return "Resets soon"
    dt_utc = ensure_utc(dt_val)
    if not dt_utc:
        return "Resets soon"

    now = datetime.now(timezone.utc)
    diff_ms = (dt_utc - now).total_seconds() * 1000

    if diff_ms <= 0:
        return "Resets soon"

    total_minutes = int(diff_ms // (1000 * 60))
    days = total_minutes // (60 * 24)
    hours = (total_minutes % (60 * 24)) // 60
    minutes = total_minutes % 60

    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    parts.append(f"{minutes}m")

    return f"resets in {' '.join(parts)}"


class QuotaService:
    @staticmethod
    def build_user_response(
        user: UserDB,
        config: AppConfigDB | None = None,
        subscription_tier: str | None = None,
        subscription_status: str | None = None,
        is_subscribed: bool = False,
    ) -> UserResponse:
        now = datetime.now(timezone.utc)
        reset_at = ensure_utc(user.reset_at)
        weekly_reset_at = ensure_utc(user.weekly_reset_at)

        is_6h_expired = reset_at is None or reset_at <= now
        is_weekly_expired = weekly_reset_at is None or weekly_reset_at <= now

        effective_6h = 0 if is_6h_expired else user.tokens_used_6h
        effective_weekly = 0 if is_weekly_expired else user.tokens_used_weekly

        limit_6h, limit_weekly = resolve_user_limits(user, config, subscription_tier)
        tier = normalize_tier(subscription_tier)

        pct_6h = min(round((effective_6h / limit_6h) * 100, 1), 100.0) if limit_6h > 0 else 0.0
        pct_weekly = min(round((effective_weekly / limit_weekly) * 100, 1), 100.0) if limit_weekly > 0 else 0.0

        return UserResponse(
            id=user.id,
            email=user.email,
            reset_at=user.reset_at,
            pct_6h=pct_6h,
            reset_countdown_6h=format_countdown_string(user.reset_at),
            weekly_reset_at=user.weekly_reset_at,
            pct_weekly=pct_weekly,
            reset_countdown_weekly=format_countdown_string(user.weekly_reset_at),
            tier=tier,
            subscription_status=subscription_status,
            is_subscribed=is_subscribed,
            custom_instructions=getattr(user, "custom_instructions", None),
        )
