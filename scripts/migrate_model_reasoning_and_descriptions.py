import asyncio
import os
import sys

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.database import create_db_client, init_db
from app.models.config import AppConfigDB


async def migrate():
    """Migration: add model_descriptions and update model_reasoning_modes with
    accurate per-model reasoning support.

    This script performs a MERGE (update) on the existing `configs/app_config`
    document so that other live fields (token limits, feature toggles, etc.) are
    preserved unchanged.

    Safe to run multiple times (idempotent).
    """
    print("Initializing Database Connection...")
    init_db()
    db = create_db_client()

    config_ref = db.collection("configs").document("app_config")
    config_doc = await config_ref.get()

    default_config = AppConfigDB()

    if not config_doc.exists:
        print("'configs/app_config' does not exist — creating full default document...")
        await config_ref.set(default_config.model_dump(mode="json"))
        print("Created 'configs/app_config' with defaults.")
        return

    existing_data = config_doc.to_dict() or {}
    print(f"\nFound existing 'configs/app_config' with {len(existing_data)} fields.")

    # Build the targeted update — only the fields this migration owns
    updates: dict = {
        # Always overwrite reasoning modes so they reflect latest model capabilities
        "model_reasoning_modes": default_config.model_reasoning_modes,
        # Merge descriptions — keep any custom overrides already in Firestore
        "model_descriptions": {
            **default_config.model_descriptions,
            # Preserve any manually customised descriptions already stored
            **(existing_data.get("model_descriptions") or {}),
        },
    }

    print("\nApplying the following updates to 'configs/app_config':")
    for key, value in updates.items():
        print(f"  [{key}]")
        if isinstance(value, dict):
            for k, v in value.items():
                print(f"      {k}: {v}")
        else:
            print(f"      {value}")

    await config_ref.update(updates)

    print("\nMigration completed successfully!")
    print("  - model_reasoning_modes: updated with per-model capability-accurate levels")
    print("  - model_descriptions: merged (existing custom values preserved)")


if __name__ == "__main__":
    asyncio.run(migrate())
