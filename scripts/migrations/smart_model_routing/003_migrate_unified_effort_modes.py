import asyncio
import os
import sys

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from app.database import create_db_client, init_db
from app.models.config import AppConfigDB, ReasoningLevel, RoutingMode


async def migrate():
    """Unify reasoning effort levels and routing policies to ['fast', 'balanced', 'extended'].

    Date: 2026-04-10
    Idempotent: Normalizes all model reasoning_modes to canonical values and invalidates Redis cache.
    """
    print("  [003_migrate_unified_effort_modes] Initializing Firestore connection...")
    init_db()
    db = create_db_client()

    config_ref = db.collection("configs").document("app_config")
    config_doc = await config_ref.get()

    default_config = AppConfigDB()

    if not config_doc.exists:
        print("  [003_migrate_unified_effort_modes] Document missing — creating default config...")
        data = default_config.model_dump(mode="json")
        await config_ref.set(data)
        return

    existing = config_doc.to_dict() or {}

    canonical_effort_levels = [
        ReasoningLevel.FAST.value,
        ReasoningLevel.BALANCED.value,
        ReasoningLevel.EXTENDED.value,
    ]

    models_list = existing.get("models_list", {})
    for model_name, default_model_cfg in default_config.models_list.items():
        if model_name in models_list:
            models_list[model_name]["reasoning_modes"] = [
                mode.value if isinstance(mode, ReasoningLevel) else str(mode)
                for mode in default_model_cfg.reasoning_modes
            ]
        else:
            models_list[model_name] = default_model_cfg.model_dump(mode="json")

    updates = {
        "allowed_reasoning_levels": canonical_effort_levels,
        "default_reasoning_level": ReasoningLevel.BALANCED.value,
        "default_routing_mode": RoutingMode.BALANCED.value,
        "models_list": models_list,
    }

    await config_ref.update(updates)
    print(
        f"  [003_migrate_unified_effort_modes] ✓ Updated {len(models_list)} models to canonical reasoning levels: {canonical_effort_levels}"
    )

    # Invalidate Redis cache if available
    try:
        from app.core.cache_keys import CacheKeys
        from app.core.redis import redis_cache

        await redis_cache.connect()
        await redis_cache.delete(CacheKeys.system_config())
        await redis_cache.disconnect()
        print("  [003_migrate_unified_effort_modes] ✓ Invalidated Redis system_config cache")
    except Exception as exc:
        print(f"  [003_migrate_unified_effort_modes] (Redis cache invalidation skipped: {exc})")


if __name__ == "__main__":
    asyncio.run(migrate())
