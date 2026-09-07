"""Firestore Composite Indexes Migration Runner
============================================
Feature: indexes / composite-indexes
Description: Cloud Firestore composite indexes defined in firestore.indexes.json
"""

import asyncio
from pathlib import Path

from scripts.deploy_indexes import deploy_indexes, get_default_database, get_default_project

FEATURE_NAME = "indexes"
FEATURE_DESCRIPTION = "Cloud Firestore composite indexes defined in firestore.indexes.json"

MIGRATION_STEPS = [
    (
        "deploy_composite_indexes",
        "2026-09-07",
        "Deploy composite indexes defined in firestore.indexes.json",
        "scripts.deploy_indexes",
    ),
]


async def run():
    print("\n=======================================================")
    print(f"🚀 Running Migrations for Feature: [{FEATURE_NAME}]")
    print(f"   {FEATURE_DESCRIPTION}")
    print("=======================================================")

    project_id = get_default_project()
    database_id = get_default_database()
    file_path = Path("firestore.indexes.json")

    loop = asyncio.get_running_loop()
    success = await loop.run_in_executor(
        None,
        deploy_indexes,
        file_path,
        project_id,
        database_id,
        False,  # dry_run
        False,  # wait
        None,  # credentials_path
    )
    if not success:
        raise RuntimeError("Failed to deploy composite indexes")


if __name__ == "__main__":
    asyncio.run(run())
