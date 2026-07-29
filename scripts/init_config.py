import asyncio
import os
import sys

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.database import create_db_client, init_db
from app.models.config import AppConfigDB


async def run_migration():
    print("Initializing Database Connection...")
    init_db()
    db = create_db_client()

    doc_ref = db.collection("configs").document("app_config")
    snapshot = await doc_ref.get()

    if snapshot.exists:
        print("Config document 'configs/app_config' already exists in Firestore.")
        data = snapshot.to_dict() or {}
        print("Current configuration:", data)
    else:
        print("Creating default config document 'configs/app_config' in Firestore...")
        default_config = AppConfigDB()
        await doc_ref.set(default_config.model_dump(mode="json"))
        print("Successfully created 'configs/app_config' in Firestore!")


if __name__ == "__main__":
    asyncio.run(run_migration())
