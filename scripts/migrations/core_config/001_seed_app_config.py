import asyncio
import os
import sys

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from app.database import create_db_client, init_db
from app.models.config import AppConfigDB


async def migrate():
    """Initial migration: seed Firestore 'configs/app_config' with default AppConfigDB.

    Date: 2026-02-15
    Idempotent: Uses merge=True so existing customizations are preserved.
    """
    print("  [001_seed_app_config] Initializing Firestore connection...")
    init_db()
    db = create_db_client()

    doc_ref = db.collection("configs").document("app_config")
    doc = await doc_ref.get()

    default_config = AppConfigDB()
    default_dict = default_config.model_dump(mode="json", exclude={"allowed_text_models"})

    if not doc.exists:
        print("  [001_seed_app_config] 'configs/app_config' missing. Creating default document...")
        await doc_ref.set(default_dict)
        print("  [001_seed_app_config] ✓ Created default config document.")
    else:
        existing = doc.to_dict() or {}
        # Fill in any top-level keys that are completely missing
        missing_keys = {k: v for k, v in default_dict.items() if k not in existing}
        if missing_keys:
            print(f"  [001_seed_app_config] Adding missing default keys: {list(missing_keys.keys())}")
            await doc_ref.update(missing_keys)
            print("  [001_seed_app_config] ✓ Updated missing keys.")
        else:
            print("  [001_seed_app_config] ✓ Config document already exists and contains all required base keys.")


if __name__ == "__main__":
    asyncio.run(migrate())
