import json
import logging
from typing import Any

import redis.asyncio as redis

from app.core.config import settings

logger = logging.getLogger(__name__)


class RedisCache:
    def __init__(self):
        self.redis_client = None

    async def connect(self):
        if not self.redis_client:
            if settings.redis_url:
                try:
                    self.redis_client = redis.from_url(settings.redis_url, decode_responses=True)
                    await self.redis_client.ping()
                    logger.info("Connected to Redis cache successfully.")
                except Exception as e:
                    logger.error(f"Failed to connect to Redis: {e}")
                    self.redis_client = None
            else:
                logger.warning("REDIS_URL not set. Caching will be disabled.")

    async def disconnect(self):
        if self.redis_client:
            await self.redis_client.aclose()
            logger.info("Disconnected from Redis cache.")

    async def get(self, key: str) -> Any | None:
        if not self.redis_client:
            return None
        try:
            val = await self.redis_client.get(key)
            if val:
                return json.loads(val)
        except Exception as e:
            logger.error(f"Redis get error for key {key}: {e}")
        return None

    async def set(self, key: str, value: Any, expire: int = 3600):
        if not self.redis_client:
            return
        try:
            if hasattr(value, "model_dump"):
                payload = value.model_dump(mode="json")
            else:
                payload = value
            await self.redis_client.set(key, json.dumps(payload, default=str), ex=expire)
        except Exception as e:
            logger.error(f"Redis set error for key {key}: {e}")

    async def delete(self, key: str):
        if not self.redis_client:
            return
        try:
            await self.redis_client.delete(key)
        except Exception as e:
            logger.error(f"Redis delete error for key {key}: {e}")

    async def delete_by_prefix(self, prefix: str):
        if not self.redis_client:
            return
        try:
            cursor = "0"
            while cursor != 0:
                cursor, keys = await self.redis_client.scan(cursor=cursor, match=f"{prefix}*", count=100)
                if keys:
                    await self.redis_client.delete(*keys)
        except Exception as e:
            logger.error(f"Redis delete_by_prefix error for prefix {prefix}: {e}")


redis_cache = RedisCache()
