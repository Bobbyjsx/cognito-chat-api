import logging

from fastapi import HTTPException
from google.cloud.firestore_v1.async_client import AsyncClient

from app.models.config import AppConfigDB

logger = logging.getLogger(__name__)


class ConfigRepository:
    def __init__(self, db: AsyncClient):
        self.db = db
        self.collection = self.db.collection("configs")

    async def get_config(self) -> AppConfigDB:
        """Fetches the global application configuration from Redis cache / Firestore.

        Raises HTTPException 500 if the app_config document has not been created by migrations.
        """
        from app.core.cache_keys import CacheKeys
        from app.core.redis import redis_cache

        try:
            cached = await redis_cache.get(CacheKeys.system_config(), model_cls=AppConfigDB)
            if cached:
                return cached
        except Exception as exc:
            logger.debug("Redis system_config get failed: %s", exc)

        doc_ref = self.collection.document("app_config")
        doc = await doc_ref.get()

        if not doc.exists:
            raise HTTPException(
                status_code=500,
                detail="System configuration 'configs/app_config' not found in database. Please run migrations.",
            )

        data = doc.to_dict() or {}
        config = AppConfigDB(**data)

        try:
            await redis_cache.set(CacheKeys.system_config(), config, expire=60)
        except Exception as exc:
            logger.debug("Redis system_config set failed: %s", exc)

        return config

    async def update_config(self, config: AppConfigDB) -> AppConfigDB:
        from app.core.cache_keys import CacheKeys
        from app.core.redis import redis_cache

        doc_ref = self.collection.document("app_config")
        data = config.model_dump(mode="json")
        await doc_ref.set(data)

        try:
            await redis_cache.set(CacheKeys.system_config(), config, expire=300)
        except Exception as exc:
            logger.debug("Redis system_config update set failed: %s", exc)

        return config
