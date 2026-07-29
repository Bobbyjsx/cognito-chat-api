import asyncio
from datetime import datetime, timedelta, timezone
import sys
import os

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from google.cloud.firestore_v1 import DELETE_FIELD
from app.database import create_db_client, init_db
from app.utils.datetime import ensure_utc


async def run_migration():
    print("Initializing Database Connection...")
    init_db()
    db = create_db_client()

    now = datetime.now(timezone.utc)
    next_6h = (now + timedelta(hours=6)).isoformat()
    next_weekly = (now + timedelta(weeks=1)).isoformat()

    users_ref = db.collection("users")
    docs = users_ref.stream()

    migrated_count = 0
    async for doc in docs:
        data = doc.to_dict() or {}
        updates = {}

        # 1. Remove legacy token_limit field
        if "token_limit" in data:
            updates["token_limit"] = DELETE_FIELD

        # 2. Ensure 6-hourly quota fields exist
        if "tokens_used_6h" not in data:
            updates["tokens_used_6h"] = 0
        if "token_limit_6h" not in data:
            updates["token_limit_6h"] = 60_000

        reset_at = ensure_utc(data.get("reset_at"))
        if reset_at is None:
            updates["reset_at"] = next_6h

        # 3. Ensure weekly quota fields exist
        if "tokens_used_weekly" not in data:
            updates["tokens_used_weekly"] = 0
        if "token_limit_weekly" not in data:
            updates["token_limit_weekly"] = 300_000

        weekly_reset_at = ensure_utc(data.get("weekly_reset_at"))
        if weekly_reset_at is None:
            updates["weekly_reset_at"] = next_weekly

        if updates:
            print(f"Migrating user document: {doc.id} with updates: {list(updates.keys())}")
            await doc.reference.update(updates)
            migrated_count += 1
        else:
            print(f"User document {doc.id} is already up to date.")

    print(f"\nMigration complete! Total user documents updated: {migrated_count}")


if __name__ == "__main__":
    asyncio.run(run_migration())
