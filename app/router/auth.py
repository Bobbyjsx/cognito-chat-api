import logging

from fastapi import APIRouter, Depends, HTTPException, status
from google.cloud.firestore_v1.async_client import AsyncClient
from pydantic import BaseModel, Field

from app.api.dependencies import get_persisted_user
from app.billing.entitlements import lookup_active_tier
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
    return QuotaService.build_user_response(user, config, subscription_tier="free")


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
    current_user=Depends(get_persisted_user),
    config_repo: ConfigRepository = Depends(get_config_repo),
    db: AsyncClient = Depends(get_db),
):
    from app.core.cache_keys import CacheKeys
    from app.core.redis import redis_cache

    cache_key = CacheKeys.user_profile(current_user.id)
    cached_data = await redis_cache.get(cache_key)
    if cached_data:
        from datetime import datetime, timezone

        from app.utils.datetime import ensure_utc

        reset_at = ensure_utc(cached_data.get("reset_at"))
        weekly_reset_at = ensure_utc(cached_data.get("weekly_reset_at"))
        now = datetime.now(timezone.utc)
        if (reset_at and reset_at <= now) or (weekly_reset_at and weekly_reset_at <= now):
            await redis_cache.delete(cache_key)
            await redis_cache.delete(CacheKeys.user_auth(current_user.id))
        else:
            return cached_data

    config = await config_repo.get_config()
    from app.billing.entitlements import get_active_tier, get_entitlement_level
    from app.billing.repository import SubscriptionRepository

    sub = await SubscriptionRepository(db).get_by_user_id(str(current_user.id))
    subscription_tier = get_active_tier(sub)
    subscription_status = sub.status.value if sub else "expired"
    is_subscribed = get_entitlement_level(sub) > 0

    response = QuotaService.build_user_response(
        current_user,
        config,
        subscription_tier=subscription_tier,
        subscription_status=subscription_status,
        is_subscribed=is_subscribed,
    )

    redis_cache.set_bg(cache_key, response.model_dump(mode="json"), expire=300)
    return response


class CustomInstructionsRequest(BaseModel):
    custom_instructions: str | None = Field(default=None, max_length=1500)


@router.put("/custom-instructions")
async def update_custom_instructions(
    request: CustomInstructionsRequest,
    current_user=Depends(get_persisted_user),
    db: AsyncClient = Depends(get_db),
):

    subscription_tier = await lookup_active_tier(db, str(current_user.id))
    if subscription_tier != "premium":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Custom instructions are exclusively available on the Premium plan.",
        )

    user_repo = UserRepository(db)
    await user_repo.update_custom_instructions(current_user.id, request.custom_instructions)
    return {
        "message": "Custom instructions updated successfully.",
        "custom_instructions": request.custom_instructions,
    }


@router.post("/refresh", response_model=TokenResponse)
async def refresh_tokens(request: RefreshRequest, db: AsyncClient = Depends(get_db)):
    from app.core.token_manager import server_token_manager

    logger.info("POST /auth/refresh: Received direct client token refresh request")
    return await server_token_manager.refresh_tokens(request.refresh_token, db)
