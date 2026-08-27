import logging

from app.core.cache_keys import CacheKeys
from app.core.redis import redis_cache

logger = logging.getLogger(__name__)


async def blacklist_model(model_id: str, duration_sec: int = 60) -> None:
    """Blacklist a model for a specific duration in seconds using Redis."""
    if redis_cache.redis_client:
        await redis_cache.set(CacheKeys.model_blacklist(model_id), "1", expire=duration_sec)
        logger.warning("Blacklisted model '%s' in Redis for %s seconds", model_id, duration_sec)
    else:
        logger.warning("Cannot blacklist model '%s' because Redis is not connected.", model_id)


async def is_blacklisted(model_id: str) -> bool:
    """Check if a model is currently blacklisted in Redis."""
    if not redis_cache.redis_client:
        return False
    val = await redis_cache.get(CacheKeys.model_blacklist(model_id))
    return val is not None


async def get_blacklisted_models(model_ids: list[str]) -> set[str]:
    """Check a batch of models."""
    if not redis_cache.redis_client:
        return set()

    blacklisted = set()
    for mid in model_ids:
        if await redis_cache.get(CacheKeys.model_blacklist(mid)):
            blacklisted.add(mid)
    return blacklisted
