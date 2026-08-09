from fastapi import APIRouter, Depends
from google.cloud.firestore_v1.async_client import AsyncClient

from app.database import get_db
from app.models.config import AppConfigDB
from app.repositories.config import ConfigRepository

router = APIRouter(prefix="/config", tags=["config"])


@router.get("", response_model=AppConfigDB)
async def get_system_config(db: AsyncClient = Depends(get_db)):
    from app.core.redis import redis_cache
    from app.core.cache_keys import CacheKeys

    cache_key = CacheKeys.system_config()
    cached_data = await redis_cache.get(cache_key)
    if cached_data:
        return cached_data

    repo = ConfigRepository(db)
    config = await repo.get_config()
    await redis_cache.set(cache_key, config.model_dump(mode="json"), expire=3600)
    return config
