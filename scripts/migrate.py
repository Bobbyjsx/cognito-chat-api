#!/usr/bin/env python3
"""Cognito Chat API — Central Database Migration CLI
=================================================

Runs database and schema migrations per feature in strict chronological order.
All migrations are designed to be idempotent (safe to run repeatedly).

Usage:
  python scripts/migrate.py <feature_name>
  python scripts/migrate.py all
  python scripts/migrate.py --list

Examples:
  python scripts/migrate.py smart-model-routing
  python scripts/migrate.py user-auth
  python scripts/migrate.py speech-to-text
  python scripts/migrate.py core-config
  python scripts/migrate.py chat-sessions
  python scripts/migrate.py all
"""

import argparse
import asyncio
import os
import sys

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.migrations.attachments import runner as attachments_runner
from scripts.migrations.chat_sessions import runner as chat_sessions_runner
from scripts.migrations.core_config import runner as core_config_runner
from scripts.migrations.smart_model_routing import runner as smart_model_routing_runner
from scripts.migrations.speech_to_text import runner as speech_to_text_runner
from scripts.migrations.user_auth import runner as user_auth_runner

# Registry of available feature migration runners
FEATURES = {
    "core-config": {
        "aliases": ["core_config", "core", "config", "initial-setup", "seed"],
        "runner": core_config_runner,
        "description": "Baseline Firestore schemas and system configuration",
    },
    "user-auth": {
        "aliases": ["user_auth", "users", "auth", "quota"],
        "runner": user_auth_runner,
        "description": "User documents, usage periods, and token quota enforcement",
    },
    "speech-to-text": {
        "aliases": ["speech_to_text", "stt", "audio"],
        "runner": speech_to_text_runner,
        "description": "Speech-to-text toggles and model selection configuration",
    },
    "smart-model-routing": {
        "aliases": ["smart_model_routing", "routing", "models", "reasoning"],
        "runner": smart_model_routing_runner,
        "description": "Structured models_list, capability scoring, and unified effort levels",
    },
    "chat-sessions": {
        "aliases": ["chat_sessions", "sessions", "session-list", "session-preview"],
        "runner": chat_sessions_runner,
        "description": "Session list documents and denormalized last_message_content previews",
    },
    "attachments": {
        "aliases": ["attachment", "files", "gcs", "storage", "upload"],
        "runner": attachments_runner,
        "description": "Canonical storage identity backfill and Firestore signed URL isolation",
    },
}


def resolve_feature(name: str):
    name_clean = name.strip().lower().replace("_", "-")
    if name_clean in FEATURES:
        return name_clean, FEATURES[name_clean]

    for feat_key, feat_val in FEATURES.items():
        if name_clean in [a.replace("_", "-") for a in feat_val["aliases"]]:
            return feat_key, feat_val
    return None, None


def list_features():
    print("\n=======================================================")
    print("📋 Available Feature Migrations")
    print("=======================================================\n")
    for feat_key, feat_val in FEATURES.items():
        runner = feat_val["runner"]
        aliases = ", ".join(feat_val["aliases"])
        print(f"🔸 Feature: {feat_key}")
        print(f"   Description: {feat_val['description']}")
        print(f"   Aliases: {aliases}")
        print("   Migration Steps:")
        for step_name, date_str, desc, _ in runner.MIGRATION_STEPS:
            print(f"     • {step_name} [{date_str}] — {desc}")
        print()


async def run_feature_migration(feature_name: str):
    _feat_key, feat_info = resolve_feature(feature_name)
    if not feat_info:
        print(f"\n❌ Error: Unknown feature '{feature_name}'.")
        print(f"Available features: {', '.join(FEATURES.keys())} (or 'all')")
        print("Run with `--list` to see all migration steps.\n")
        sys.exit(1)

    await feat_info["runner"].run()


async def run_all_migrations():
    print("\n=======================================================")
    print("🚀 Running ALL Feature Migrations in Order")
    print("=======================================================")
    for feat_info in FEATURES.values():
        await feat_info["runner"].run()
    print("\n🎉 ALL database migrations executed successfully!\n")


def main():
    parser = argparse.ArgumentParser(
        description="Cognito Chat API Feature Database Migrations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "feature",
        nargs="?",
        default="all",
        help="Feature name to migrate (e.g. 'smart-model-routing', 'user-auth', 'all')",
    )
    parser.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="List all features and their migration steps",
    )

    args = parser.parse_args()

    if args.list:
        list_features()
        return

    if args.feature.lower() in ("all", "run-all"):
        asyncio.run(run_all_migrations())
    else:
        asyncio.run(run_feature_migration(args.feature))


if __name__ == "__main__":
    main()
