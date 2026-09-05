import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from app.database import create_db_client, init_db
from app.models.config import AppConfigDB
from app.services.chats import AgentService

_BATCH_SIZE = 400


async def migrate():
    """Backfill missing session titles in Firestore and ensure AI title config exists in configs/app_config.

    Date: 2026-09-05
    Idempotent: Only writes when title is missing/empty, and preserves existing config keys.
    """
    print("  [002_backfill_session_titles_and_config] Initializing Firestore connection...")
    init_db()
    db = create_db_client()

    # 1. Ensure configs/app_config has title generation keys
    config_ref = db.collection("configs").document("app_config")
    config_doc = await config_ref.get()
    defaults = AppConfigDB()

    if config_doc.exists:
        existing_cfg = config_doc.to_dict() or {}
        cfg_updates = {}
        if "enable_ai_title_generation" not in existing_cfg:
            cfg_updates["enable_ai_title_generation"] = defaults.enable_ai_title_generation
        if "title_generation_model" not in existing_cfg:
            cfg_updates["title_generation_model"] = defaults.title_generation_model

        if cfg_updates:
            print(f"  [002_backfill_session_titles_and_config] Updating configs/app_config: {cfg_updates}")
            await config_ref.update(cfg_updates)
            print("  [002_backfill_session_titles_and_config] ✓ App config updated with title generation settings.")
        else:
            print(
                "  [002_backfill_session_titles_and_config] ✓ Title generation keys already present in configs/app_config."
            )
    else:
        print("  [002_backfill_session_titles_and_config] Creating default configs/app_config...")
        await config_ref.set(defaults.model_dump(mode="json"))

    # 2. Backfill missing session titles
    scanned = 0
    updated = 0
    batch = db.batch()
    pending = 0

    async for doc in db.collection("sessions").stream():
        scanned += 1
        data = doc.to_dict() or {}
        title = data.get("title")

        if title and str(title).strip():
            continue

        preview = data.get("last_message_content") or data.get("first_message_preview") or ""
        if preview:
            backfilled_title, _ = AgentService._evaluate_title_strategy(str(preview))
        else:
            backfilled_title = "New Chat"

        batch.update(doc.reference, {"title": backfilled_title})
        pending += 1
        updated += 1

        if pending >= _BATCH_SIZE:
            await batch.commit()
            print(f"  [002_backfill_session_titles_and_config] Committed {updated} updates ({scanned} scanned)")
            batch = db.batch()
            pending = 0

    if pending:
        await batch.commit()

    print(f"  [002_backfill_session_titles_and_config] ✓ Backfilled {updated} session titles (scanned {scanned}).")

    try:
        from app.core.redis import redis_cache

        await redis_cache.connect()
        await redis_cache.delete_by_prefix("sessions:")
        await redis_cache.delete_by_prefix("config:")
        await redis_cache.disconnect()
        print("  [002_backfill_session_titles_and_config] ✓ Invalidated Redis session & config caches.")
    except Exception as exc:
        print(f"  [002_backfill_session_titles_and_config] (Redis cache invalidation skipped: {exc})")


if __name__ == "__main__":
    asyncio.run(migrate())
