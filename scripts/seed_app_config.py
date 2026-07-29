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
    config = AppConfigDB()
    data = config.model_dump(mode="json")

    print("Writing/updating system configuration document 'configs/app_config' in Firestore...")
    await doc_ref.set(data)
    print("Migration successful! System configuration set to:")
    for key, val in data.items():
        print(f"  - {key}: {val}")


if __name__ == "__main__":
    asyncio.run(run_migration())
