#!/usr/bin/env python3
"""Cognito Chat API — Deploy Cloud Firestore Composite Indexes
===========================================================

Idempotently deploys composite indexes defined in `firestore.indexes.json`
to Google Cloud Firestore using the Firestore Admin API.

Features:
- Reads index definitions from `firestore.indexes.json`.
- Queries existing indexes from Firestore to skip already created or building indexes.
- Resolves credentials via GOOGLE_APPLICATION_CREDENTIALS, FIREBASE_CREDENTIALS_PATH,
  or Application Default Credentials (ADC).
- Supports configurable project and database (e.g. `cognito-prod-db`).
- Supports `--dry-run` to preview planned operations without mutating Firestore.
- Supports `--wait` to optionally block until all indexes finish building.

Usage:
  uv run python scripts/deploy_indexes.py
  uv run python scripts/deploy_indexes.py --dry-run
  uv run python scripts/deploy_indexes.py --database cognito-prod-db --project project-atlas-501612
  uv run python scripts/deploy_indexes.py --file firestore.indexes.json --wait
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from google.api_core.exceptions import AlreadyExists, GoogleAPICallError
from google.cloud.firestore_admin_v1 import FirestoreAdminClient, types
from google.oauth2 import service_account


def get_default_project(credentials_path: str | None = None) -> str:
    """Resolve default GCP project ID from env or credentials file."""
    if os.environ.get("GOOGLE_CLOUD_PROJECT"):
        return os.environ["GOOGLE_CLOUD_PROJECT"]

    cred_path = (
        credentials_path
        or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        or os.environ.get("FIREBASE_CREDENTIALS_PATH")
    )
    if cred_path and os.path.exists(cred_path):
        try:
            with open(cred_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "project_id" in data:
                    return data["project_id"]
        except (json.JSONDecodeError, OSError):
            pass

    return "project-atlas-501612"


def get_default_database() -> str:
    """Resolve Firestore database ID from env or fallback to cognito-prod-db."""
    return os.environ.get("FIRESTORE_DATABASE") or "cognito-prod-db"


def create_admin_client(credentials_path: str | None = None) -> FirestoreAdminClient:
    """Initialize FirestoreAdminClient with appropriate credentials."""
    cred_file = (
        credentials_path
        or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        or os.environ.get("FIREBASE_CREDENTIALS_PATH")
    )
    if cred_file and os.path.exists(cred_file):
        try:
            creds = service_account.Credentials.from_service_account_file(cred_file)
            return FirestoreAdminClient(credentials=creds)
        except Exception as e:
            print(f"⚠️  Warning: Failed to load service account credentials from {cred_file}: {e}")
            print("   Falling back to Application Default Credentials...")

    return FirestoreAdminClient()


def load_indexes_file(filepath: Path) -> list[dict[str, Any]]:
    """Load and parse the indexes definition file."""
    if not filepath.exists():
        raise FileNotFoundError(f"Indexes configuration file not found at: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    indexes = data.get("indexes", [])
    if not isinstance(indexes, list):
        raise TypeError(f"Invalid format in {filepath}: 'indexes' must be a list.")

    return indexes


def normalize_target_index(idx: dict[str, Any]) -> tuple[str, str, tuple[tuple[str, str | None, str | None], ...]]:
    """Normalize a target index dict from firestore.indexes.json into a hashable signature."""
    col = idx.get("collectionGroup", "")
    query_scope = idx.get("queryScope", "COLLECTION")
    fields = []
    for f in idx.get("fields", []):
        order = f.get("order")
        array_config = f.get("arrayConfig")
        fields.append((f.get("fieldPath", ""), order, array_config))
    return (col, query_scope, tuple(fields))


def normalize_existing_index(idx: types.Index) -> tuple[str, str, tuple[tuple[str, str | None, str | None], ...]]:
    """Normalize a live types.Index proto into a hashable signature."""
    # idx.name format: projects/{project}/databases/{database}/collectionGroups/{collection}/indexes/{index_id}
    parts = idx.name.split("/")
    col = parts[parts.index("collectionGroups") + 1] if "collectionGroups" in parts else ""
    query_scope = idx.query_scope.name if hasattr(idx.query_scope, "name") else str(idx.query_scope)
    fields = []
    for f in idx.fields:
        order = (
            f.order.name
            if hasattr(f.order, "name") and f.order != types.Index.IndexField.Order.ORDER_UNSPECIFIED
            else None
        )
        array_config = (
            f.array_config.name
            if hasattr(f.array_config, "name")
            and f.array_config != types.Index.IndexField.ArrayConfig.ARRAY_CONFIG_UNSPECIFIED
            else None
        )
        fields.append((f.field_path, order, array_config))
    return (col, query_scope, tuple(fields))


def fetch_existing_indexes(
    client: FirestoreAdminClient,
    project_id: str,
    database_id: str,
    collection_groups: set[str],
) -> list[types.Index]:
    """Fetch existing composite indexes from Firestore."""
    # First attempt database-wide wildcard listing
    try:
        parent = client.collection_group_path(project_id, database_id, "-")
        return list(client.list_indexes(parent=parent))
    except Exception as e:
        print(f"ℹ️  Wildcard collection index listing failed ({e}). Falling back to per-collection listing...")

    all_indexes: list[types.Index] = []
    for col in collection_groups:
        try:
            parent = client.collection_group_path(project_id, database_id, col)
            all_indexes.extend(list(client.list_indexes(parent=parent)))
        except Exception as col_err:
            print(f"⚠️  Could not list indexes for collection group '{col}': {col_err}")

    return all_indexes


def build_index_proto(idx: dict[str, Any]) -> types.Index:
    """Convert index dictionary to google.cloud.firestore_admin_v1 types.Index proto."""
    query_scope_str = idx.get("queryScope", "COLLECTION")
    query_scope = getattr(types.Index.QueryScope, query_scope_str, types.Index.QueryScope.COLLECTION)

    fields = []
    for f in idx.get("fields", []):
        field_kwargs: dict[str, Any] = {"field_path": f["fieldPath"]}
        if "order" in f:
            order_enum = getattr(types.Index.IndexField.Order, f["order"], None)
            if order_enum is not None:
                field_kwargs["order"] = order_enum
        if "arrayConfig" in f:
            array_enum = getattr(types.Index.IndexField.ArrayConfig, f["arrayConfig"], None)
            if array_enum is not None:
                field_kwargs["array_config"] = array_enum
        fields.append(types.Index.IndexField(**field_kwargs))

    return types.Index(query_scope=query_scope, fields=fields)


def format_index_desc(col: str, fields: list[dict[str, Any]]) -> str:
    """Format index description for user-friendly logging."""
    field_parts = []
    for f in fields:
        path = f.get("fieldPath", "")
        mode = f.get("order") or f.get("arrayConfig") or ""
        field_parts.append(f"{path} ({mode})" if mode else path)
    return f"{col} [{', '.join(field_parts)}]"


def deploy_indexes(
    file_path: Path,
    project_id: str,
    database_id: str,
    dry_run: bool = False,
    wait: bool = False,
    credentials_path: str | None = None,
) -> bool:
    """Deploy composite indexes from JSON file to Firestore."""
    print("\n=======================================================")
    print("🔥 Cloud Firestore Composite Index Deployment")
    print("=======================================================")
    print(f"  • Source file:  {file_path}")
    print(f"  • Project ID:   {project_id}")
    print(f"  • Database ID:  {database_id}")
    print(f"  • Dry run:      {dry_run}")
    print(f"  • Wait build:   {wait}")
    print("=======================================================\n")

    if os.environ.get("FIRESTORE_EMULATOR_HOST"):
        print("ℹ️  FIRESTORE_EMULATOR_HOST detected. Composite index deployment is skipped in emulator environment.")
        return True

    target_indexes = load_indexes_file(file_path)
    print(f"📋 Loaded {len(target_indexes)} index definition(s) from {file_path.name}")

    if not target_indexes:
        print("✓ No indexes defined in file. Nothing to deploy.")
        return True

    collection_groups = {idx["collectionGroup"] for idx in target_indexes if "collectionGroup" in idx}

    client = create_admin_client(credentials_path)

    print("\n🔍 Querying existing indexes from Firestore...")
    try:
        existing_indexes = fetch_existing_indexes(client, project_id, database_id, collection_groups)
        print(f"✓ Found {len(existing_indexes)} existing composite index(es) in database '{database_id}'\n")
    except Exception as e:
        print(f"❌ Failed to query existing indexes: {e}")
        return False

    existing_signatures: set[tuple[str, str, tuple[tuple[str, str | None, str | None], ...]]] = {
        normalize_existing_index(idx) for idx in existing_indexes
    }

    operations = []
    already_present_count = 0
    created_count = 0

    print("🚀 Processing indexes:")
    for idx_dict in target_indexes:
        col = idx_dict.get("collectionGroup", "")
        fields = idx_dict.get("fields", [])
        desc = format_index_desc(col, fields)
        sig = normalize_target_index(idx_dict)

        if sig in existing_signatures:
            already_present_count += 1
            print(f"  • [EXISTS]  {desc}")
            continue

        if dry_run:
            created_count += 1
            print(f"  • [PLAN]    {desc} (would create)")
            continue

        parent = client.collection_group_path(project_id, database_id, col)
        index_proto = build_index_proto(idx_dict)

        try:
            print(f"  • [CREATING] {desc} ...")
            operation = client.create_index(parent=parent, index=index_proto)
            op_name = getattr(operation, "operation", None)
            op_id = getattr(op_name, "name", "initiated")
            print(f"    ↳ Successfully initiated index creation: {op_id}")
            operations.append((desc, operation))
            created_count += 1
        except AlreadyExists:
            already_present_count += 1
            print("    ↳ Already exists in Firestore. Skipped.")
        except GoogleAPICallError as api_err:
            print(f"    ❌ API Error creating index for {desc}: {api_err}")
            return False
        except Exception as err:
            print(f"    ❌ Unexpected error creating index for {desc}: {err}")
            return False

    if wait and operations:
        print(f"\n⏳ Waiting for {len(operations)} index build operation(s) to finish...")
        for desc, op in operations:
            try:
                op.result()
                print(f"  ✓ [READY] {desc}")
            except Exception as wait_err:
                print(f"  ❌ Error waiting for {desc}: {wait_err}")
                return False

    print("\n=======================================================")
    print("📊 Deployment Summary")
    print("=======================================================")
    print(f"  • Total defined:       {len(target_indexes)}")
    print(f"  • Already up to date:  {already_present_count}")
    print(f"  • Created / Initiated: {created_count}")
    print("=======================================================\n")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Deploy Cloud Firestore composite indexes from JSON definition file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-f",
        "--file",
        type=Path,
        default=Path("firestore.indexes.json"),
        help="Path to indexes JSON file (default: firestore.indexes.json)",
    )
    parser.add_argument(
        "-p",
        "--project",
        type=str,
        default=None,
        help="Google Cloud Project ID (default: $GOOGLE_CLOUD_PROJECT or project from credentials)",
    )
    parser.add_argument(
        "-d",
        "--database",
        type=str,
        default=None,
        help="Cloud Firestore Database ID (default: $FIRESTORE_DATABASE or cognito-prod-db)",
    )
    parser.add_argument(
        "-c",
        "--credentials",
        type=str,
        default=None,
        help="Path to service account JSON credentials file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview actions without modifying Cloud Firestore",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for index building operations to complete before exiting",
    )

    args = parser.parse_args()

    project_id = args.project or get_default_project(args.credentials)
    database_id = args.database or get_default_database()

    success = deploy_indexes(
        file_path=args.file,
        project_id=project_id,
        database_id=database_id,
        dry_run=args.dry_run,
        wait=args.wait,
        credentials_path=args.credentials,
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
