"""Attachments & File Storage Migrations
======================================
Feature: attachments / files / storage
Description: Backfill canonical storage identity (bucket + object_name) and purge any transient signed URLs from Firestore.

Migrations in order:
  1. 001_backfill_storage_identity_and_clean_urls (Date: 2026-09-05):
     Backfills bucket & object_name from storage_uri in attachments and message parts,
     and removes any transient signed URLs (url, url_expires_at) from Firestore documents.
"""

from __future__ import annotations

import asyncio
import importlib

FEATURE_NAME = "attachments"
FEATURE_DESCRIPTION = "Canonical storage identity backfill and Firestore signed URL isolation"

MIGRATION_STEPS = [
    (
        "001_backfill_storage_identity_and_clean_urls",
        "2026-09-05",
        "Backfill bucket/object_name and purge signed URLs from attachments, messages, and shared chats",
        "scripts.migrations.attachments.001_backfill_storage_identity_and_clean_urls",
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
