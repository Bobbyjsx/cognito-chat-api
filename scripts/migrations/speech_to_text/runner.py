"""Speech-to-Text (STT) Migrations
=================================
Feature: speech-to-text / stt
Description: Speech-to-text configuration toggles and audio model selection.

Migrations in order:
  1. 001_migrate_stt_config (Date: 2026-03-01):
     Adds `enable_ai_stt` and `stt_model` to `configs/app_config`.
"""

import asyncio
import importlib

FEATURE_NAME = "speech-to-text"
FEATURE_DESCRIPTION = "Speech-to-text configuration and model settings"

MIGRATION_STEPS = [
    (
        "001_migrate_stt_config",
        "2026-03-01",
        "Add enable_ai_stt and stt_model fields to configs/app_config",
        "scripts.migrations.speech_to_text.001_migrate_stt_config",
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
