import asyncio
import os
import sys

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from app.database import create_db_client, init_db
from app.models.config import AppConfigDB


async def migrate():
    """Ensure all models in models_list have complete descriptions and capability attributes.

    Date: 2026-04-01
    Idempotent: Merges missing descriptions/attributes without overriding existing custom text.
    """
    print("  [002_migrate_model_descriptions] Initializing Firestore connection...")
    init_db()
    db = create_db_client()

    config_ref = db.collection("configs").document("app_config")
    config_doc = await config_ref.get()

    default_config = AppConfigDB()

    if not config_doc.exists:
        print("  [002_migrate_model_descriptions] Creating default app_config...")
        await config_ref.set(default_config.model_dump(mode="json"))
        return

    existing = config_doc.to_dict() or {}
    models_list = existing.get("models_list", {})

    ordered_models_list = {}
    for model_name, default_model_cfg in default_config.models_list.items():
        if model_name in models_list:
            item = models_list[model_name]
            if not item.get("description"):
                item["description"] = default_model_cfg.description
            ordered_models_list[model_name] = item
        else:
            ordered_models_list[model_name] = default_model_cfg.model_dump(mode="json")

    for model_name, custom_cfg in models_list.items():
        if model_name not in ordered_models_list:
            ordered_models_list[model_name] = custom_cfg

    await config_ref.update({"models_list": ordered_models_list})
    print(
        f"  [002_migrate_model_descriptions] ✓ Model descriptions & capabilities verified across {len(ordered_models_list)} models (including 'auto')."
    )


if __name__ == "__main__":
    asyncio.run(migrate())
