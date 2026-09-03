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


async def get_current_user(
    request: Request,
    response: Response,
    token: str = Depends(oauth2_scheme),
    db: AsyncClient = Depends(get_db),
) -> UserDB:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    payload = None
    refresh_token = request.headers.get("x-refresh-token") or request.headers.get("X-Refresh-Token")

    try:
        payload = await decode_jwt_payload(token)
        # Check if token is near expiration and refresh proactively if refresh token is supplied
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
                logger.info(
                    "get_current_user: Proactive token refresh successful. Injected updated token headers for '%s'.",
                    request.url.path,
                )
            except Exception as ref_exc:
                logger.debug("Proactive token refresh encountered non-critical error: %s", ref_exc)
    except (jwt.ExpiredSignatureError, PyJWTError) as token_err:
        # Access token is expired or invalid — attempt transparent server-side refresh if refresh token is present
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
                logger.info(
                    "get_current_user: Transparent server-side token refresh successful. Injected updated token headers for '%s'.",
                    request.url.path,
                )
            except Exception as ref_fail:
                logger.warning(
                    "get_current_user: Transparent server-side token refresh failed on path '%s': %s",
                    request.url.path,
                    ref_fail,
                )
                raise credentials_exception
        else:
            logger.debug(
                "get_current_user: Access token invalid/expired on path '%s' and no x-refresh-token provided.",
                request.url.path,
            )
            raise credentials_exception

    if payload is None:
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

    # 1. Check Redis cache first for fast user resolution
    from app.core.cache_keys import CacheKeys
    from app.core.redis import redis_cache

    try:
        cached_user = await redis_cache.get(CacheKeys.user_auth(user_id), model_cls=UserDB)
        if cached_user:
            return cached_user
    except Exception as exc:
        logger.debug("Redis user cache check failed: %s", exc)

    # 2. Fetch fresh from Firestore
    user = await UserRepository(db).get_by_id(user_id)

    if user is None:
        email = payload.get("email")
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

    # Cache user auth for 120 seconds
    try:
        await redis_cache.set(CacheKeys.user_auth(user_id), user, expire=120)
    except Exception as exc:
        logger.debug("Failed to cache user in Redis: %s", exc)

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
