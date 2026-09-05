"""Chat Session Migrations
========================
Feature: chat-sessions / sessions
Description: Session list document shape used by GET /agent/sessions.

Migrations in order:
  1. 001_trim_last_message_content (Date: 2026-09-04):
     Trims denormalized session.last_message_content to the list preview length.
"""

import asyncio
import importlib

FEATURE_NAME = "chat-sessions"
FEATURE_DESCRIPTION = "Chat session list documents and denormalized preview fields"

MIGRATION_STEPS = [
    (
        "001_trim_last_message_content",
        "2026-09-04",
        "Trim session.last_message_content to the list preview length",
        "scripts.migrations.chat_sessions.001_trim_last_message_content",
    ),
    (
        "002_backfill_session_titles_and_config",
        "2026-09-05",
        "Backfill missing session titles and populate AI title generation settings in app_config",
        "scripts.migrations.chat_sessions.002_backfill_session_titles_and_config",
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
