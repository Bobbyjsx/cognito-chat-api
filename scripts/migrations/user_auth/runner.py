"""User Authentication & Quota Migrations
========================================
Feature: user-auth / users
Description: User document schema upgrades, usage tracking periods, and token quota enforcement.

Migrations in order:
  1. 001_migrate_users_schema (Date: 2026-03-10):
     Migrates user documents to 6-hourly and weekly token usage tracking.
  2. 002_migrate_quota_limits (Date: 2026-03-20):
     Enforces global quota defaults in `configs/app_config` and cleans legacy limit fields from user documents.
"""

import asyncio
import importlib

FEATURE_NAME = "user-auth"
FEATURE_DESCRIPTION = "User authentication schemas, token tracking, and quota management"

MIGRATION_STEPS = [
    (
        "001_migrate_users_schema",
        "2026-03-10",
        "Migrate user documents to 6-hourly and weekly usage tracking schema",
        "scripts.migrations.user_auth.001_migrate_users_schema",
    ),
    (
        "002_migrate_quota_limits",
        "2026-03-20",
        "Enforce global quota limits and remove user-level limit overrides",
        "scripts.migrations.user_auth.002_migrate_quota_limits",
    ),
]


async def run():
    print("\n=======================================================")
    print(f"🚀 Running Migrations for Feature: [{FEATURE_NAME}]")
    print(f"   {FEATURE_DESCRIPTION}")
    print("=======================================================")

    for step_name, date_str, desc, module_path in MIGRATION_STEPS:
        print(f"\n▶ Step: {step_name} (Date: {date_str})")
        print(f"  Description: {desc}")
        module = importlib.import_module(module_path)
        if hasattr(module, "migrate"):
            await module.migrate()
        elif hasattr(module, "run_migration"):
            await module.run_migration()

    print(f"\n✅ All migrations for [{FEATURE_NAME}] completed successfully!\n")


if __name__ == "__main__":
    asyncio.run(run())
