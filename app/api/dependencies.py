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
            headers={"User-Agent": "Mozilla/5.0 (compatible; cognito-chat-api)"}
        )
    return _jwks_client


def get_provider(request: Request) -> BaseProvider:
    """Return the shared AI provider instance (created at app startup)."""
    return request.app.state.provider


def get_tool_registry(request: Request) -> ToolRegistry:
    """Return the shared tool registry (created at app startup)."""
    return request.app.state.tool_registry


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
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    payload = None

    # 1. Try JWKS / Identity Service token verification
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
    except Exception:
        pass

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
    user = await UserRepository(db).get_by_id(user_id)
    if user is None:
        # Auto-provision user record in Firestore for Identity Service users
        email = payload.get("email")
        if not email and settings.identity_service_url:
            import httpx
            try:
                # Need to verify exactly what the identity service URL should be. 
                # If settings.identity_service_url is e.g. http://localhost:8002
                me_url = f"{settings.identity_service_url.rstrip('/')}/api/v1/auth/me"
                async with httpx.AsyncClient() as client:
                    resp = await client.get(
                        me_url,
                        headers={"Authorization": f"Bearer {token}", "User-Agent": "Mozilla/5.0 (compatible; cognito-chat-api)"}
                    )
                    if resp.status_code == 200:
                        email = resp.json().get("email")
            except Exception as e:
                import logging
                logging.error(f"Failed to fetch user email from identity service: {e}")
                
        email = email or f"{user_id}@auth.identity"
        new_user = UserDB(
            id=user_id,
            email=email,
            hashed_password="",
        )
        user = await UserRepository(db).create(new_user)

    return user

