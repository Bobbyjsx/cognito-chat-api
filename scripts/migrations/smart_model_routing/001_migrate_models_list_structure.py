import asyncio
import os
import sys

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from google.cloud.firestore_v1 import DELETE_FIELD

from app.database import create_db_client, init_db
from app.models.config import AppConfigDB

# Fields being replaced by models_list
STALE_FIELDS = [
    "allowed_text_models",
    "model_reasoning_modes",
    "model_descriptions",
]


async def migrate():
    """Migrate flat text model fields into structured models_list and delete legacy fields.

    Date: 2026-03-25
    Idempotent: Safe to run multiple times; preserves existing custom overrides.
    """
    print("  [001_migrate_models_list_structure] Initializing Firestore connection...")
    init_db()
    db = create_db_client()

    config_ref = db.collection("configs").document("app_config")
    config_doc = await config_ref.get()

    default_config = AppConfigDB()

    if not config_doc.exists:
        print("  [001_migrate_models_list_structure] 'configs/app_config' does not exist — creating defaults...")
        data = default_config.model_dump(mode="json")
        await config_ref.set(data)
        print("  [001_migrate_models_list_structure] ✓ Created default config.")
        return

    existing = config_doc.to_dict() or {}

    # Build models_list from canonical defaults
    new_models_list: dict = {name: cfg.model_dump(mode="json") for name, cfg in default_config.models_list.items()}

    # Merge existing overrides
    existing_models_list = existing.get("models_list", {})
    for model_name, existing_cfg in existing_models_list.items():
        if model_name in new_models_list:
            new_models_list[model_name].update({k: v for k, v in existing_cfg.items() if k in ("enabled",)})
        else:
            new_models_list[model_name] = existing_cfg

    stale_found = [f for f in STALE_FIELDS if f in existing]

    updates: dict = {"models_list": new_models_list}
    for field in stale_found:
        updates[field] = DELETE_FIELD

    await config_ref.update(updates)
    print(f"  [001_migrate_models_list_structure] ✓ models_list updated ({len(new_models_list)} models).")
    if stale_found:
        print(f"  [001_migrate_models_list_structure] ✓ Removed stale fields: {stale_found}")


if __name__ == "__main__":
    asyncio.run(migrate())
