from __future__ import annotations

import asyncio
import logging
import time

import httpx
import jwt
from jwt import PyJWKSet

from app.core.cache_keys import CacheKeys
from app.core.config import settings
from app.core.redis import redis_cache

logger = logging.getLogger(__name__)

JWKS_TTL_SECONDS = 3600
IDENTITY_ALGORITHMS = ("EdDSA", "RS256", "ES256")

_memory_jwks: dict | None = None
_memory_loaded_at = 0.0
_lock = asyncio.Lock()


def identity_jwks_url() -> str:
    url = settings.identity_jwks_url or ""
    if not url:
        base = (settings.identity_service_url or "").rstrip("/")
        if not base:
            return ""
        url = f"{base}/.well-known/jwks.json"
    if "localhost" in url or "127.0.0.1" in url:
        return ""
    return url


def _memory_fresh() -> bool:
    return bool(_memory_jwks) and (time.time() - _memory_loaded_at) < JWKS_TTL_SECONDS


def _store_memory(jwks: dict) -> dict:
    global _memory_jwks, _memory_loaded_at
    _memory_jwks = jwks
    _memory_loaded_at = time.time()
    return jwks


async def prefetch_jwks() -> dict | None:
    logger.info("Waking identity service to prefetch JWKS...")
    try:
        result = await get_jwks()
        logger.info("Identity service wake finished.")
        return result
    except Exception as exc:
        logger.info("Identity service wake finished (with ignored error): %s", exc)
        return None


async def get_jwks() -> dict | None:
    if _memory_fresh():
        return _memory_jwks

    async with _lock:
        if _memory_fresh():
            return _memory_jwks

        cached = await redis_cache.get(CacheKeys.identity_jwks())
        if isinstance(cached, dict) and cached.get("keys"):
            return _store_memory(cached)

        url = identity_jwks_url()
        if not url:
            return None

        timeout = httpx.Timeout(20.0, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url, headers={"User-Agent": "cognito-chat-api"})
            response.raise_for_status()
            jwks = response.json()

        if not isinstance(jwks, dict) or not jwks.get("keys"):
            return None

        _store_memory(jwks)
        try:
            await redis_cache.set(CacheKeys.identity_jwks(), jwks, expire=JWKS_TTL_SECONDS)
        except Exception as exc:
            logger.debug("Failed to cache JWKS in Redis: %s", exc)
        return jwks


async def decode_identity_jwt(token: str) -> dict:
    jwks = await get_jwks()
    if not jwks:
        raise jwt.PyJWTError("JWKS unavailable")

    header = jwt.get_unverified_header(token)
    kid = header.get("kid")
    key_set = PyJWKSet.from_dict(jwks)
    if kid:
        signing_key = key_set[kid]
    elif key_set.keys:
        signing_key = key_set.keys[0]
    else:
        raise jwt.PyJWTError("JWKS has no keys")
    return jwt.decode(
        token,
        signing_key.key,
        algorithms=list(IDENTITY_ALGORITHMS),
        options={"verify_aud": False},
    )
