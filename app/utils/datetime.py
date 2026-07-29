from datetime import datetime, timezone
from typing import Any


def ensure_utc(dt_val: Any) -> datetime | None:
    """Safely converts string, datetime, or Firestore Timestamp to a UTC-aware datetime object."""
    if dt_val is None:
        return None
    if isinstance(dt_val, str):
        dt = datetime.fromisoformat(dt_val)
    elif hasattr(dt_val, "to_datetime"):
        dt = dt_val.to_datetime()
    elif isinstance(dt_val, datetime):
        dt = dt_val
    else:
        return None

    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
