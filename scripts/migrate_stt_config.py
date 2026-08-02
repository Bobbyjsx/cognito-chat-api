import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.database import create_db_client, init_db
from app.models.config import AppConfigDB


async def migrate():
    """Add enable_ai_stt and stt_model fields to configs/app_config.
    Idempotent — skips fields that already exist.
    """
    print("Initializing Database Connection...")
    init_db()
    db = create_db_client()

    config_ref = db.collection("configs").document("app_config")
    config_doc = await config_ref.get()
    defaults = AppConfigDB()

    if not config_doc.exists:
        print("Document not found — creating full default config...")
        await config_ref.set(defaults.model_dump(mode="json"))
        return

    existing = config_doc.to_dict() or {}
    updates = {}

    if "enable_ai_stt" not in existing:
        updates["enable_ai_stt"] = defaults.enable_ai_stt
    if "stt_model" not in existing:
        updates["stt_model"] = defaults.stt_model

    if updates:
        print(f"Applying updates: {updates}")
        await config_ref.update(updates)
        print("Migration successful!")
    else:
        print("Fields already exist — nothing to do.")


if __name__ == "__main__":
    asyncio.run(migrate())
