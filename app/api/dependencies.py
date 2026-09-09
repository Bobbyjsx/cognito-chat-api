import logging

import jwt
from fastapi import Depends, Header, HTTPException, Request, Response, status
from fastapi.security import OAuth2PasswordBearer
from google.cloud.firestore_v1.async_client import AsyncClient
from jwt import PyJWTError

from app.core.config import settings
from app.core.jwks import IDENTITY_ALGORITHMS, decode_identity_jwt
from app.core.token_manager import server_token_manager
from app.database import get_db
from app.integrations.cloud_tasks import CloudTasksDispatcher
from app.models.users import UserDB
from app.providers.base import BaseProvider
from app.repositories.users import UserRepository
from app.storage.base import StorageBackend
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")


def get_provider_registry(request: Request):
    """Return the shared ProviderRegistry instance (created at app startup)."""
    if hasattr(request.app.state, "provider_registry"):
        return request.app.state.provider_registry
    from app.providers.registry import create_default_provider_registry

    return create_default_provider_registry()


def get_provider(request: Request) -> BaseProvider:
    """Return the shared default AI provider instance (created at app startup)."""
    if hasattr(request.app.state, "provider"):
        return request.app.state.provider
    return get_provider_registry(request).get("gemini")


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


def get_attachment_url_service(
    storage: StorageBackend = Depends(get_storage_backend),
):
    """Return an AttachmentUrlService configured with the current storage backend."""
    from app.services.attachment_url import AttachmentUrlService

    return AttachmentUrlService(storage)


def _email_from_payload(payload: dict, user_id: str) -> str:
    email = payload.get("email")
    if email:
        return email
    import re

    safe_local_part = re.sub(r"[^a-zA-Z0-9._-]", "_", str(user_id))
    return f"{safe_local_part}@auth.identity"


def _user_from_payload(payload: dict) -> UserDB:
    user_id = payload.get("sub")
    if not user_id or payload.get("type") == "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    import uuid

    try:
        user_id = str(uuid.UUID(str(user_id)))
    except ValueError:
        user_id = str(user_id)
    return UserDB(id=user_id, email=_email_from_payload(payload, user_id), hashed_password="")


async def decode_jwt_payload(token: str) -> dict:
    header = jwt.get_unverified_header(token)
    alg = header.get("alg") or ""
    if alg in IDENTITY_ALGORITHMS:
        return await decode_identity_jwt(token)
    return jwt.decode(
        token,
        settings.secret_key,
        algorithms=[settings.algorithm, "HS256"],
        options={"verify_aud": False},
    )


async def _authenticate_payload(
    request: Request,
    response: Response,
    token: str,
    db: AsyncClient,
) -> dict:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    payload = None
    refresh_token = request.headers.get("x-refresh-token") or request.headers.get("X-Refresh-Token")

    try:
        payload = await decode_jwt_payload(token)
        exp = payload.get("exp")
        if exp and server_token_manager.is_near_expiry(exp) and refresh_token:
            logger.info(
                "get_current_user: Access token near expiry on path '%s'. Initiating proactive token refresh.",
                request.url.path,
            )
            try:
                refreshed = await server_token_manager.refresh_tokens(refresh_token, db)
                response.headers["X-New-Access-Token"] = refreshed.access_token
                response.headers["X-New-Refresh-Token"] = refreshed.refresh_token
            except Exception as ref_exc:
                logger.debug("Proactive token refresh encountered non-critical error: %s", ref_exc)
    except (jwt.ExpiredSignatureError, PyJWTError) as token_err:
        if refresh_token:
            logger.info(
                "get_current_user: Access token expired/invalid (%s) on path '%s'. Initiating transparent server-side refresh.",
                type(token_err).__name__,
                request.url.path,
            )
            try:
                refreshed = await server_token_manager.refresh_tokens(refresh_token, db)
                response.headers["X-New-Access-Token"] = refreshed.access_token
                response.headers["X-New-Refresh-Token"] = refreshed.refresh_token
                payload = await decode_jwt_payload(refreshed.access_token)
            except Exception as ref_fail:
                logger.warning(
                    "get_current_user: Transparent server-side token refresh failed on path '%s': %s",
                    request.url.path,
                    ref_fail,
                )
                raise credentials_exception
        else:
            raise credentials_exception

    if payload is None:
        raise credentials_exception
    return payload


async def get_current_user(
    request: Request,
    response: Response,
    token: str = Depends(oauth2_scheme),
    db: AsyncClient = Depends(get_db),
) -> UserDB:
    """Resolve identity from the JWT. Does not round-trip Firestore/Redis.

    Use ``get_persisted_user`` when quota, profile, or JIT provisioning is required.
    """
    payload = await _authenticate_payload(request, response, token, db)
    return _user_from_payload(payload)


async def get_persisted_user(
    current_user: UserDB = Depends(get_current_user),
    db: AsyncClient = Depends(get_db),
) -> UserDB:
    """Load the Firestore user document (Redis-backed) and JIT-provision if missing."""
    from app.core.cache_keys import CacheKeys
    from app.core.redis import redis_cache

    user_id = str(current_user.id)
    try:
        cached_user = await redis_cache.get(CacheKeys.user_auth(user_id), model_cls=UserDB)
        if cached_user:
            from datetime import datetime, timezone

            from app.utils.datetime import ensure_utc

            now = datetime.now(timezone.utc)
            reset_at = ensure_utc(cached_user.reset_at)
            weekly_reset_at = ensure_utc(cached_user.weekly_reset_at)
            if (reset_at and reset_at <= now) or (weekly_reset_at and weekly_reset_at <= now):
                await redis_cache.delete(CacheKeys.user_auth(user_id))
            else:
                return cached_user
    except Exception as exc:
        logger.debug("Redis user cache check failed: %s", exc)

    user = await UserRepository(db).get_by_id(user_id)
    if user is None:
        user = await UserRepository(db).create(
            UserDB(id=user_id, email=current_user.email, hashed_password=""),
        )

    redis_cache.set_bg(CacheKeys.user_auth(user_id), user, expire=120)
    return user


def get_tasks_dispatcher(request: Request) -> CloudTasksDispatcher | None:
    """Returns the shared CloudTasksDispatcher instance if worker_provider is 'cloudtasks', otherwise None."""
    if settings.worker_provider.lower() != "cloudtasks":
        return None
    dispatcher = getattr(request.app.state, "tasks_dispatcher", None)
    if dispatcher is None:
        dispatcher = CloudTasksDispatcher(
            project=settings.cloud_tasks_project,
            location=settings.cloud_tasks_location,
            queue=settings.cloud_tasks_queue,
            worker_url=settings.cloud_tasks_worker_url,
            service_account_email=settings.cloud_tasks_service_account_email,
        )
    return dispatcher


async def verify_cloud_tasks_caller(
    request: Request,
    x_cloudtasks_queuename: str | None = Header(None, alias="X-CloudTasks-QueueName"),
) -> bool:
    """
    Validates that a task request originates from Google Cloud Tasks or an authorized internal worker.
    """
    # Cloud Tasks injects specific HTTP headers
    if x_cloudtasks_queuename:
        return True

    # Allow in local development
    if settings.debug or getattr(settings, "environment", "development") == "development":
        return True

    # We could also verify OIDC token here if we used one

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Unauthorized Cloud Tasks invocation",
    )


async def get_optional_current_user(
    request: Request,
    response: Response,
    db: AsyncClient = Depends(get_db),
) -> UserDB | None:
    """
    Safely resolves the current user if an Authorization Bearer header is present.
    Returns None if unauthenticated or if the token is invalid/expired without throwing.
    """
    auth_header = request.headers.get("Authorization") or request.headers.get("authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return None

    token = auth_header.split(" ", 1)[1].strip()
    if not token:
        return None

    try:
        return await get_current_user(request=request, response=response, token=token, db=db)
    except Exception:
        return None
