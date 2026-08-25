import logging

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from google.cloud.firestore_v1.async_client import AsyncClient
from jwt import PyJWKClient, PyJWTError

from app.core.config import settings
from app.database import get_db
from app.models.users import UserDB
from app.providers.base import BaseProvider
from app.repositories.users import UserRepository
from app.storage.base import StorageBackend
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

_jwks_client: PyJWKClient | None = None


def get_jwks_client() -> PyJWKClient | None:
    global _jwks_client
    if _jwks_client is None and settings.identity_service_url:
        jwks_url = settings.identity_jwks_url or f"{settings.identity_service_url.rstrip('/')}/.well-known/jwks.json"
        _jwks_client = PyJWKClient(
            jwks_url,
            cache_jwk_set=True,
            lifespan=3600,
            headers={"User-Agent": "Mozilla/5.0 (compatible; cognito-chat-api)"},
        )
    return _jwks_client


def get_provider(request: Request) -> BaseProvider:
    """Return the shared AI provider instance (created at app startup)."""
    return request.app.state.provider


def get_tool_registry(request: Request) -> ToolRegistry:
    """Return the shared tool registry (created at app startup)."""
    return request.app.state.tool_registry


def get_smart_router(request: Request):
    """Return the shared smart model router instance."""
    from app.ai.router import SmartModelRouter

    return getattr(request.app.state, "smart_router", None) or SmartModelRouter()


def get_storage_backend() -> StorageBackend:
    """Return the application-wide storage backend."""
    return build_storage_backend_instance()


_storage_backend: StorageBackend | None = None


def build_storage_backend_instance() -> StorageBackend:
    global _storage_backend
    if _storage_backend is None:
        from app.storage import build_storage_backend

        _storage_backend = build_storage_backend()
    return _storage_backend


async def get_current_user(token: str = Depends(oauth2_scheme), db: AsyncClient = Depends(get_db)) -> UserDB:
    import time

    auth_start = time.perf_counter()

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    payload = None

    # 1. Try JWKS / Identity Service token verification
    jwks_start = time.perf_counter()
    try:
        jwk_client = get_jwks_client()
        if jwk_client:
            signing_key = jwk_client.get_signing_key_from_jwt(token)
            payload = jwt.decode(
                token,
                signing_key.key,
                algorithms=["EdDSA", "RS256", "ES256", "HS256"],
                options={"verify_aud": False},
            )
            logger.debug("[PERF][Auth] JWKS verification succeeded in %.2f ms", (time.perf_counter() - jwks_start) * 1000.0)
    except Exception as exc:
        logger.debug("[PERF][Auth] JWKS token verification bypassed in %.2f ms: %s", (time.perf_counter() - jwks_start) * 1000.0, exc)

    # 2. Fallback to symmetric secret key verification (local dev / tests)
    if payload is None:
        try:
            payload = jwt.decode(
                token,
                settings.secret_key,
                algorithms=[settings.algorithm, "HS256"],
                options={"verify_aud": False},
            )
        except PyJWTError:
            raise credentials_exception

    user_id: str | None = payload.get("sub")
    token_type: str | None = payload.get("type")
    if not user_id or token_type == "refresh":
        raise credentials_exception

    import uuid

    try:
        # Normalize to standard dashed UUID string
        user_id = str(uuid.UUID(user_id))
    except ValueError:
        pass

    # Always fetch fresh from Firestore so tokens_used reflects live state
    db_user_start = time.perf_counter()
    user = await UserRepository(db).get_by_id(user_id)
    logger.debug("[PERF][Auth] User DB lookup took %.2f ms", (time.perf_counter() - db_user_start) * 1000.0)

    if user is None:
        # Auto-provision user record in Firestore for Identity Service users
        email = payload.get("email")
        if not email and settings.identity_service_url:
            import httpx

            try:
                me_url = f"{settings.identity_service_url.rstrip('/')}/api/v1/auth/me"
                async with httpx.AsyncClient(timeout=1.5) as client:
                    resp = await client.get(
                        me_url,
                        headers={
                            "Authorization": f"Bearer {token}",
                            "User-Agent": "Mozilla/5.0 (compatible; cognito-chat-api)",
                        },
                    )
                    if resp.status_code == 200:
                        email = resp.json().get("email")
            except Exception as e:
                logger.debug("Failed to fetch user email from identity service (skipped): %s", e)

        if not email:
            import re

            safe_local_part = re.sub(r"[^a-zA-Z0-9._-]", "_", str(user_id))
            email = f"{safe_local_part}@auth.identity"

        new_user = UserDB(
            id=user_id,
            email=email,
            hashed_password="",
        )
        user = await UserRepository(db).create(new_user)

    total_auth_ms = (time.perf_counter() - auth_start) * 1000.0
    logger.info("[PERF][Auth] get_current_user completed in %.2f ms (user_id=%s)", total_auth_ms, user_id)
    return user
