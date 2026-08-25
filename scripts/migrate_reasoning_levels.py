import asyncio
import os
import sys

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.database import create_db_client, init_db
from app.models.config import AppConfigDB, ReasoningLevel


async def migrate():
    """Migration: update allowed_reasoning_levels and per-model reasoning_modes in Firestore configs/app_config."""
    print("Initializing Database Connection...")
    init_db()
    db = create_db_client()

    config_ref = db.collection("configs").document("app_config")
    config_doc = await config_ref.get()

    default_config = AppConfigDB()

    if not config_doc.exists:
        print("'configs/app_config' does not exist — creating full default document...")
        data = default_config.model_dump(mode="json")
        await config_ref.set(data)
        print("Created 'configs/app_config' with defaults.")
        return

    existing = config_doc.to_dict() or {}
    print(f"\nFound existing 'configs/app_config' with {len(existing)} fields.")

    canonical_reasoning_levels = [
        ReasoningLevel.NONE.value,
        ReasoningLevel.MINIMAL.value,
        ReasoningLevel.LOW.value,
        ReasoningLevel.MEDIUM.value,
        ReasoningLevel.HIGH.value,
    ]

    # Update models_list reasoning_modes from canonical defaults
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
        "allowed_reasoning_levels": canonical_reasoning_levels,
        "default_reasoning_level": ReasoningLevel.MEDIUM.value,
        "models_list": models_list,
    }

    print("\n── Applying Reasoning Level Updates ──")
    print(f"  allowed_reasoning_levels: {canonical_reasoning_levels}")
    print(f"  default_reasoning_level: {ReasoningLevel.MEDIUM.value}")
    print("  models_list reasoning_modes:")
    for name, cfg in models_list.items():
        print(f"    - {name}: {cfg.get('reasoning_modes')}")

    await config_ref.update(updates)

    # Invalidate Redis cache
    from app.core.cache_keys import CacheKeys
    from app.core.redis import redis_cache

    try:
        await redis_cache.connect()
        await redis_cache.delete(CacheKeys.system_config())
        await redis_cache.disconnect()
        print("\n  ✓ Invalidated Redis system_config cache")
    except Exception as exc:
        print(f"\n  (Note: Redis cache invalidation skipped: {exc})")

    print("\nMigration completed successfully!")


if __name__ == "__main__":
    asyncio.run(migrate())
