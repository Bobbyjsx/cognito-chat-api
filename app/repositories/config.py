import logging
import time

from fastapi import HTTPException
from google.cloud.firestore_v1.async_client import AsyncClient

from app.models.config import AppConfigDB

logger = logging.getLogger(__name__)

_CONFIG_MEMORY_TTL_SECONDS = 30
_config_memory: AppConfigDB | None = None
_config_loaded_at = 0.0


def clear_config_memory() -> None:
    global _config_memory, _config_loaded_at
    _config_memory = None
    _config_loaded_at = 0.0


class ConfigRepository:
    def __init__(self, db: AsyncClient):
        self.db = db
        self.collection = self.db.collection("configs")

    async def get_config(self) -> AppConfigDB:
        """Fetches the global application configuration from memory / Redis / Firestore.

        Raises HTTPException 500 if the app_config document has not been created by migrations.
        """
        global _config_memory, _config_loaded_at

        if _config_memory is not None and (time.time() - _config_loaded_at) < _CONFIG_MEMORY_TTL_SECONDS:
            return _config_memory

        from app.core.cache_keys import CacheKeys
        from app.core.redis import redis_cache

        try:
            cached = await redis_cache.get(CacheKeys.system_config(), model_cls=AppConfigDB)
            if cached:
                _config_memory = cached
                _config_loaded_at = time.time()
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
        _config_memory = config
        _config_loaded_at = time.time()

        try:
            await redis_cache.set(CacheKeys.system_config(), config, expire=60)
        except Exception as exc:
            logger.debug("Redis system_config set failed: %s", exc)

        return config

    async def update_config(self, config: AppConfigDB) -> AppConfigDB:
        global _config_memory, _config_loaded_at
        from app.core.cache_keys import CacheKeys
        from app.core.redis import redis_cache

        doc_ref = self.collection.document("app_config")
        data = config.model_dump(mode="json")
        await doc_ref.set(data)

        _config_memory = config
        _config_loaded_at = time.time()

        try:
            await redis_cache.set(CacheKeys.system_config(), config, expire=300)
        except Exception as exc:
            logger.debug("Redis system_config update set failed: %s", exc)

        return config
