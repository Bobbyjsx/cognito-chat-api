"""Billing Tier Quota Migrations
================================
Feature: billing-quotas
Description: Set per-tier token quota limits for Go and Premium subscription plans.
             These limits are applied at runtime from the subscription record;
             this migration sets them as named constants in app_config and
             backfills any existing active subscribers.

Migrations in order:
  1. set_tier_quotas (Date: 2026-09-10):
     Writes Go and Premium token limits into configs/app_config.
"""

import asyncio
import importlib

FEATURE_NAME = "billing-quotas"
FEATURE_DESCRIPTION = "Token quota limits for Go and Premium subscription tiers"

MIGRATION_STEPS = [
    (
        "set_tier_quotas",
        "2026-09-10",
        "Write Go and Premium token limits into configs/app_config",
        "scripts.migrations.billing_quotas.set_tier_quotas",
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
