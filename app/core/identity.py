"""Identity-service profile helpers."""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

_SYNTHETIC_DOMAINS = frozenset({"auth.identity"})
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def is_usable_email(value: Any) -> bool:
    """Paystack (and most processors) reject synthetic / TLD-less addresses."""
    if not isinstance(value, str):
        return False
    email = value.strip()
    if not _EMAIL_RE.match(email):
        return False
    domain = email.rsplit("@", 1)[1].lower()
    return domain not in _SYNTHETIC_DOMAINS and not domain.endswith(".identity")


def email_from_mapping(data: Any) -> str | None:
    if not isinstance(data, dict):
        return None
    for key in ("email", "email_address", "emailAddress", "preferred_username"):
        candidate = data.get(key)
        if is_usable_email(candidate):
            return str(candidate).strip()
    emails = data.get("emails")
    if isinstance(emails, list):
        for item in emails:
            if is_usable_email(item):
                return str(item).strip()
            if isinstance(item, dict):
                nested = email_from_mapping(item)
                if nested:
                    return nested
    user = data.get("user")
    if isinstance(user, dict):
        return email_from_mapping(user)
    return None


def email_from_jwt_payload(payload: dict, user_id: str) -> str:
    found = email_from_mapping(payload)
    if found:
        return found
    import re as _re

    safe_local = _re.sub(r"[^a-zA-Z0-9._-]", "_", str(user_id))
    return f"{safe_local}@auth.identity"


async def fetch_identity_email(access_token: str) -> str | None:
    token = (access_token or "").strip()
    base = (settings.identity_service_url or "").rstrip("/")
    if not token or not base:
        return None

    url = f"{base}/api/v1/auth/me"
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(url, headers={"Authorization": f"Bearer {token}"})
    except httpx.HTTPError as exc:
        logger.warning("Identity profile lookup failed: %s", exc)
        return None

    if resp.status_code != 200:
        logger.warning("Identity profile lookup status=%s", resp.status_code)
        return None

    try:
        data = resp.json()
    except Exception:
        return None
    return email_from_mapping(data)
