"""Smart Model Routing Migrations
===============================
Feature: smart-model-routing
Description: Structured models_list configuration, capability scores, descriptions, and unified effort levels.

Migrations in order:
  1. 001_migrate_models_list_structure (Date: 2026-03-25):
     Refactors text model configs into `models_list` and removes stale legacy flat fields.
  2. 002_migrate_model_descriptions (Date: 2026-04-01):
     Verifies and populates complete model descriptions and capability score metadata.
  3. 003_migrate_unified_effort_modes (Date: 2026-04-10):
     Unifies reasoning effort levels and routing policies to ['fast', 'balanced', 'extended'] and clears Redis cache.
"""

import asyncio
import importlib

FEATURE_NAME = "smart-model-routing"
FEATURE_DESCRIPTION = "Smart model routing, structured models_list, descriptions, and unified reasoning effort levels"

MIGRATION_STEPS = [
    (
        "001_migrate_models_list_structure",
        "2026-03-25",
        "Refactor text model configs into structured models_list and clean stale flat fields",
        "scripts.migrations.smart_model_routing.001_migrate_models_list_structure",
    ),
    (
        "002_migrate_model_descriptions",
        "2026-04-01",
        "Populate and verify model descriptions and capability scores",
        "scripts.migrations.smart_model_routing.002_migrate_model_descriptions",
    ),
    (
        "003_migrate_unified_effort_modes",
        "2026-04-10",
        "Unify reasoning effort levels to ['fast', 'balanced', 'extended'] and invalidate Redis cache",
        "scripts.migrations.smart_model_routing.003_migrate_unified_effort_modes",
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
