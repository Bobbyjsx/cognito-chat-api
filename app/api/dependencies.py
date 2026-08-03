import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from google.cloud.firestore_v1.async_client import AsyncClient

from app.core.config import settings
from app.database import get_db
from app.models.users import UserDB
from app.providers.base import BaseProvider
from app.repositories.users import UserRepository
from app.storage.base import StorageBackend
from app.tools.registry import ToolRegistry

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")


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
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        user_id: str = payload.get("sub")
        token_type: str = payload.get("type")
        if user_id is None or token_type == "refresh":
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception

    # Always fetch fresh from Firestore so tokens_used reflects live state
    user = await UserRepository(db).get_by_id(user_id)
    if user is None:
        raise credentials_exception
    return user
