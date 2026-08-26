import logging

from fastapi import APIRouter, Depends, status
from google.cloud.firestore_v1.async_client import AsyncClient

from app.api.dependencies import get_current_user
from app.database import get_db
from app.models.users import (
    LoginRequest,
    PasswordResetRequest,
    RefreshRequest,
    TokenResponse,
    UserCreate,
    UserResponse,
)
from app.repositories.config import ConfigRepository
from app.repositories.users import UserRepository
from app.services.auth import AuthService
from app.services.quota import QuotaService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


def get_auth_service(db: AsyncClient = Depends(get_db)) -> AuthService:
    user_repo = UserRepository(db)
    return AuthService(user_repo)


def get_config_repo(db: AsyncClient = Depends(get_db)) -> ConfigRepository:
    return ConfigRepository(db)


@router.post("/signup", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def signup(
    user_data: UserCreate,
    auth_service: AuthService = Depends(get_auth_service),
    config_repo: ConfigRepository = Depends(get_config_repo),
):
    user = await auth_service.register_user(user_data)
    config = await config_repo.get_config()
    return QuotaService.build_user_response(user, config)


@router.post("/login", response_model=TokenResponse)
async def login(
    request: LoginRequest,
    auth_service: AuthService = Depends(get_auth_service),
):
    return await auth_service.login(request.email, request.password)


@router.post("/reset-password", status_code=status.HTTP_200_OK)
async def reset_password(request: PasswordResetRequest, auth_service: AuthService = Depends(get_auth_service)):
    await auth_service.change_password(request.email, request.new_password)
    return {"message": "Password updated successfully."}


@router.get("/me", response_model=UserResponse)
async def get_my_profile(
    current_user=Depends(get_current_user),
    config_repo: ConfigRepository = Depends(get_config_repo),
):
    from app.core.cache_keys import CacheKeys
    from app.core.redis import redis_cache

    cache_key = CacheKeys.user_profile(current_user.id)
    cached_data = await redis_cache.get(cache_key)
    if cached_data:
        return cached_data

    config = await config_repo.get_config()
    response = QuotaService.build_user_response(current_user, config)

    await redis_cache.set(cache_key, response.model_dump(mode="json"), expire=300)
    return response


@router.post("/refresh", response_model=TokenResponse)
async def refresh_tokens(request: RefreshRequest, db: AsyncClient = Depends(get_db)):
    from app.core.token_manager import server_token_manager

    logger.info("POST /auth/refresh: Received direct client token refresh request")
    return await server_token_manager.refresh_tokens(request.refresh_token, db)
