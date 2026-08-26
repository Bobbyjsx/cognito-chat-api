"""Core Configuration Migrations
==============================
Feature: core-config / initial-setup
Description: Core global application configuration and baseline Firestore database schemas.

Migrations in order:
  1. 001_seed_app_config (Date: 2026-02-15):
     Seeds `configs/app_config` with initial AppConfigDB baseline.
"""

import asyncio
import importlib

FEATURE_NAME = "core-config"
FEATURE_DESCRIPTION = "Core global application configuration baseline"

MIGRATION_STEPS = [
    (
        "001_seed_app_config",
        "2026-02-15",
        "Seed Firestore 'configs/app_config' with default AppConfigDB values",
        "scripts.migrations.core_config.001_seed_app_config",
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
