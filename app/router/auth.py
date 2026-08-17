from fastapi import APIRouter, Depends
from google.cloud.firestore_v1.async_client import AsyncClient

from app.api.dependencies import get_current_user
from app.database import get_db
from app.models.users import UserResponse
from app.repositories.config import ConfigRepository
from app.services.quota import QuotaService

router = APIRouter(prefix="/auth", tags=["auth"])

def get_config_repo(db: AsyncClient = Depends(get_db)) -> ConfigRepository:
    return ConfigRepository(db)

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
