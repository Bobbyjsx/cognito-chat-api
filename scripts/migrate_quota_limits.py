import asyncio
import os
import sys

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from google.cloud.firestore_v1 import DELETE_FIELD

from app.database import create_db_client, init_db
from app.models.config import AppConfigDB


async def migrate_global_config_and_users():
    """Migration script:
    1. Confirms existing fields in `configs/app_config`. Preserves any custom settings.
    2. Ensures default global quota limits (`default_token_limit_6h` and `default_token_limit_weekly`) exist in `configs/app_config`.
    3. Removes limit fields (`token_limit_6h`, `token_limit_weekly`, legacy `token_limit`) from user documents
       without modifying their used token counts (`tokens_used`, `tokens_used_6h`, `tokens_used_weekly`).
    """
    print("Initializing Database Connection...")
    init_db()
    db = create_db_client()

    config_ref = db.collection("configs").document("app_config")
    config_doc = await config_ref.get()

    print("\n--- Phase 1: Checking & Updating Global App Config ---")
    default_config = AppConfigDB()

    if config_doc.exists:
        existing_data = config_doc.to_dict() or {}
        print("Existing 'configs/app_config' document found:")
        for k, v in existing_data.items():
            print(f"  - {k}: {v}")

        # Add missing default limit fields without modifying existing custom configuration
        config_updates = {}
        if "default_token_limit_6h" not in existing_data:
            config_updates["default_token_limit_6h"] = default_config.default_token_limit_6h
        if "default_token_limit_weekly" not in existing_data:
            config_updates["default_token_limit_weekly"] = default_config.default_token_limit_weekly

        if config_updates:
            print(f"Applying missing global quota limits to app_config: {config_updates}")
            await config_ref.update(config_updates)
        else:
            print("Global app_config already contains token limits. No updates required for app_config.")
    else:
        print("'configs/app_config' does not exist. Creating default AppConfigDB document...")
        initial_data = default_config.model_dump(mode="json")
        await config_ref.set(initial_data)
        print("Created 'configs/app_config' with default limits:")
        for k, v in initial_data.items():
            print(f"  - {k}: {v}")

    print("\n--- Phase 2: Cleaning Per-User Token Limits ---")
    users_ref = db.collection("users")
    users_stream = users_ref.stream()

    migrated_users_count = 0
    async for user_doc in users_stream:
        user_data = user_doc.to_dict() or {}
        user_updates = {}

        # Remove explicit limit fields if present, retaining usage counters
        if "token_limit_6h" in user_data:
            user_updates["token_limit_6h"] = DELETE_FIELD
        if "token_limit_weekly" in user_data:
            user_updates["token_limit_weekly"] = DELETE_FIELD
        if "token_limit" in user_data:
            user_updates["token_limit"] = DELETE_FIELD

        if user_updates:
            print(f"Removing user-level limits from user {user_doc.id}: {list(user_updates.keys())}")
            await user_doc.reference.update(user_updates)
            migrated_users_count += 1
        else:
            print(f"User {user_doc.id} has no explicit user-level limits set.")

    print(f"\nMigration successfully completed! Total user documents updated: {migrated_users_count}\n")


if __name__ == "__main__":
    asyncio.run(migrate_global_config_and_users())
