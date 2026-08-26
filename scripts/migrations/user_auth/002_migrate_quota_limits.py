import asyncio
import os
import sys

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from google.cloud.firestore_v1 import DELETE_FIELD

from app.database import create_db_client, init_db
from app.models.config import AppConfigDB


async def migrate():
    """Enforce centralized quota limits in configs/app_config and clean per-user limits.

    Date: 2026-03-20
    Idempotent: Sets missing global limits and deletes user-level limit overrides.
    """
    print("  [002_migrate_quota_limits] Initializing Firestore connection...")
    init_db()
    db = create_db_client()

    config_ref = db.collection("configs").document("app_config")
    config_doc = await config_ref.get()

    default_config = AppConfigDB()

    if config_doc.exists:
        existing_data = config_doc.to_dict() or {}
        config_updates = {}
        if "default_token_limit_6h" not in existing_data:
            config_updates["default_token_limit_6h"] = default_config.default_token_limit_6h
        if "default_token_limit_weekly" not in existing_data:
            config_updates["default_token_limit_weekly"] = default_config.default_token_limit_weekly

        if config_updates:
            print(f"  [002_migrate_quota_limits] Adding global quota limits: {list(config_updates.keys())}")
            await config_ref.update(config_updates)
        else:
            print("  [002_migrate_quota_limits] ✓ Global app_config quota limits already present.")
    else:
        print("  [002_migrate_quota_limits] Creating default app_config...")
        await config_ref.set(default_config.model_dump(mode="json"))

    # Clean up user-level limit overrides
    users_ref = db.collection("users")
    users_stream = users_ref.stream()

    cleaned_users_count = 0
    async for user_doc in users_stream:
        user_data = user_doc.to_dict() or {}
        user_updates = {}

        for limit_field in ("token_limit_6h", "token_limit_weekly", "token_limit"):
            if limit_field in user_data:
                user_updates[limit_field] = DELETE_FIELD

        if user_updates:
            await user_doc.reference.update(user_updates)
            cleaned_users_count += 1

    print(f"  [002_migrate_quota_limits] ✓ Cleaned user-level limit overrides from {cleaned_users_count} users.")


if __name__ == "__main__":
    asyncio.run(migrate())
