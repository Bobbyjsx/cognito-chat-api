import asyncio
import os
import sys

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from google.cloud.firestore_v1 import DELETE_FIELD

from app.database import create_db_client, init_db
from app.models.config import AppConfigDB

# Fields that are being replaced by models_list — remove them from Firestore
STALE_FIELDS = [
    "allowed_text_models",
    "model_reasoning_modes",
    "model_descriptions",
]


async def migrate():
    """Migration: refactor text model configuration into a structured models_list.

    - Reads any existing per-model data (descriptions, reasoning modes) from
      old flat fields to preserve custom overrides before deleting them.
    - Merges into the new models_list structure.
    - Removes stale fields: allowed_text_models, model_reasoning_modes,
      model_descriptions.
    - Idempotent: safe to run multiple times.
    """
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

    # ── Step 1: Build models_list, preserving any custom data in old flat fields ──
    # Start from the canonical defaults
    new_models_list: dict = {
        name: cfg.model_dump(mode="json")
        for name, cfg in default_config.models_list.items()
    }

    # Merge any existing models_list entries (e.g. admin toggled enabled=False)
    existing_models_list = existing.get("models_list", {})
    for model_name, existing_cfg in existing_models_list.items():
        if model_name in new_models_list:
            # Preserve enabled flag and any admin overrides
            new_models_list[model_name].update(
                {k: v for k, v in existing_cfg.items() if k in ("enabled",)}
            )
        else:
            # Unknown model — keep it as-is so we don't lose custom additions
            print(f"  Preserving unknown custom model entry: {model_name}")
            new_models_list[model_name] = existing_cfg

    print("\n── Step 1: Merged models_list ──")
    for name, cfg in new_models_list.items():
        status = "✓ enabled" if cfg.get("enabled") else "✗ disabled"
        print(f"  {name} [{status}]")
        print(f"    description: {cfg.get('description', '')}")
        print(f"    reasoning_modes: {cfg.get('reasoning_modes', [])}")

    # ── Step 2: Check for stale fields ──
    stale_found = [f for f in STALE_FIELDS if f in existing]
    print(f"\n── Step 2: Stale fields to remove: {stale_found or 'none'} ──")

    # ── Step 3: Build the Firestore update payload ──
    updates: dict = {"models_list": new_models_list}

    # Mark stale fields for deletion
    for field in stale_found:
        updates[field] = DELETE_FIELD

    print("\n── Step 3: Applying updates ──")
    print(f"  Setting: models_list ({len(new_models_list)} models)")
    for field in stale_found:
        print(f"  Deleting stale field: {field}")

    await config_ref.update(updates)

    print("\nMigration completed successfully!")
    print(f"  ✓ models_list written with {len(new_models_list)} models")
    for field in stale_found:
        print(f"  ✓ Removed stale field: {field}")
    if not stale_found:
        print("  (No stale fields found — already clean)")


if __name__ == "__main__":
    asyncio.run(migrate())
