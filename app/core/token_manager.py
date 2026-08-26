import asyncio
import logging
import time

from fastapi import HTTPException
from google.cloud.firestore_v1.async_client import AsyncClient

from app.core.config import settings
from app.models.users import TokenResponse
from app.repositories.users import UserRepository
from app.services.auth import AuthService

logger = logging.getLogger(__name__)

# Configurable safety threshold in seconds (default: 60s)
TOKEN_REFRESH_BUFFER_SECONDS = 60


class ServerTokenRefreshManager:
    """Manages server-side token validation, near-expiry refresh,
    and concurrency-safe single-flight deduplication on FastAPI.
    """

    def __init__(self):
        self._locks: dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()
        # Short-lived cache (10 seconds) for freshly refreshed tokens to handle bursts
        self._recent_refreshes: dict[str, tuple[TokenResponse, float]] = {}

    async def _get_lock(self, key: str) -> asyncio.Lock:
        async with self._global_lock:
            if key not in self._locks:
                self._locks[key] = asyncio.Lock()
            return self._locks[key]

    def is_near_expiry(self, exp_timestamp: float | None, buffer_seconds: int = TOKEN_REFRESH_BUFFER_SECONDS) -> bool:
        if not exp_timestamp:
            return True
        now = time.time()
        return now >= (exp_timestamp - buffer_seconds)

    async def refresh_tokens(self, refresh_token: str, db: AsyncClient) -> TokenResponse:
        """Executes a single-flight concurrency-safe refresh operation.
        If multiple concurrent requests arrive with the same expired or near-expiry token,
        only one refresh operation executes, and all callers receive the updated tokens.
        """
        if not refresh_token or not refresh_token.strip():
            raise HTTPException(status_code=401, detail="Missing refresh token")

        key = refresh_token.strip()

        # Check recent refresh burst cache (5s TTL)
        now = time.time()
        cached = self._recent_refreshes.get(key)
        if cached:
            cached_resp, cached_time = cached
            if now - cached_time < 5.0:
                return cached_resp

        lock = await self._get_lock(key)
        async with lock:
            # Double-check burst cache inside lock
            cached = self._recent_refreshes.get(key)
            if cached:
                cached_resp, cached_time = cached
                if time.time() - cached_time < 5.0:
                    return cached_resp

            # Try local auth service refresh first
            user_repo = UserRepository(db)
            auth_service = AuthService(user_repo)

            token_response = None
            try:
                token_response = await auth_service.refresh_tokens(refresh_token)
            except Exception as local_exc:
                logger.debug("Local auth refresh did not succeed, checking identity service: %s", local_exc)

            # If local refresh failed and identity service URL is configured, try identity service
            if token_response is None and settings.identity_service_url:
                import httpx

                refresh_url = f"{settings.identity_service_url.rstrip('/')}/api/v1/auth/refresh"
                try:
                    async with httpx.AsyncClient(timeout=5.0) as client:
                        resp = await client.post(
                            refresh_url,
                            json={"refresh_token": refresh_token, "refreshToken": refresh_token},
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            token_response = TokenResponse(
                                access_token=data.get("access_token") or data.get("accessToken"),
                                refresh_token=data.get("refresh_token") or data.get("refreshToken") or refresh_token,
                                expires_in=data.get("expires_in") or data.get("expiresIn") or 1800,
                            )
                except Exception as id_exc:
                    logger.warning("Identity service token refresh failed: %s", id_exc)

            if token_response is None:
                raise HTTPException(
                    status_code=401,
                    detail="Invalid or expired refresh token",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            # Store in burst cache
            self._recent_refreshes[key] = (token_response, time.time())
            # Clean up old entries
            self._cleanup_cache()

            return token_response

    def _cleanup_cache(self):
        now = time.time()
        keys_to_delete = [k for k, (_, t) in self._recent_refreshes.items() if now - t > 30.0]
        for k in keys_to_delete:
            del self._recent_refreshes[k]


server_token_manager = ServerTokenRefreshManager()
