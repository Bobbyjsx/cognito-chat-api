from datetime import datetime, timezone

from app.models.config import AppConfigDB
from app.models.users import UserDB, UserResponse
from app.utils.datetime import ensure_utc


def resolve_user_limits(user: UserDB, config: AppConfigDB | None = None) -> tuple[int, int]:
    limit_6h = (
        user.token_limit_6h
        if user.token_limit_6h is not None
        else (config.default_token_limit_6h if config else 60_000)
    )
    limit_weekly = (
        user.token_limit_weekly
        if user.token_limit_weekly is not None
        else (config.default_token_limit_weekly if config else 300_000)
    )
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
    def build_user_response(user: UserDB, config: AppConfigDB | None = None) -> UserResponse:
        now = datetime.now(timezone.utc)
        reset_at = ensure_utc(user.reset_at)
        weekly_reset_at = ensure_utc(user.weekly_reset_at)

        is_6h_expired = reset_at is None or reset_at <= now
        is_weekly_expired = weekly_reset_at is None or weekly_reset_at <= now

        effective_6h = 0 if is_6h_expired else user.tokens_used_6h
        effective_weekly = 0 if is_weekly_expired else user.tokens_used_weekly

        limit_6h, limit_weekly = resolve_user_limits(user, config)

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
        )
