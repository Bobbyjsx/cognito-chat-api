import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from google.cloud.firestore_v1 import DELETE_FIELD

from app.database import create_db_client, init_db

_BATCH_SIZE = 400


def _parse_storage_identity(storage_uri: str | None) -> tuple[str | None, str | None]:
    """Derive canonical bucket and object_name from storage_uri."""
    if not storage_uri:
        return None, None
    if storage_uri.startswith("gs://"):
        parts = storage_uri[5:].split("/", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
    elif storage_uri.startswith("local://"):
        return "local", storage_uri[8:]
    return None, storage_uri


def _clean_file_part(part: dict) -> tuple[dict, bool]:
    """Clean transient URLs and ensure canonical storage identity in a file part.

    Returns (cleaned_part, has_changed).
    """
    changed = False
    cleaned = dict(part)

    # 1. Purge transient signed URLs from Firestore
    for key in ("url", "url_expires_at", "urlExpiresAt"):
        if key in cleaned:
            del cleaned[key]
            changed = True

    # 2. Backfill canonical storage identity if missing
    storage_uri = cleaned.get("storage_uri")
    bucket = cleaned.get("bucket")
    object_name = cleaned.get("object_name") or cleaned.get("objectName")

    if (not bucket or not object_name) and storage_uri:
        derived_bucket, derived_object = _parse_storage_identity(storage_uri)
        if derived_bucket and not bucket:
            cleaned["bucket"] = derived_bucket
            changed = True
        if derived_object and not object_name:
            cleaned["object_name"] = derived_object
            changed = True

    # 3. Standardize contentType alias
    if not cleaned.get("contentType") and cleaned.get("mediaType"):
        cleaned["contentType"] = cleaned["mediaType"]
        changed = True

    return cleaned, changed


async def migrate():
    """Backfill canonical storage identity and purge transient signed URLs.

    Date: 2026-09-05
    Idempotent: Only updates documents where fields are missing or transient URLs are present.
    """
    print("  [001_backfill_storage_identity_and_clean_urls] Initializing Firestore connection...")
    init_db()
    db = create_db_client()

    # ── 1. Migrate attachments collection ──────────────────────────────────────
    print("  [001_backfill_storage_identity_and_clean_urls] Scanning attachments collection...")
    att_scanned = 0
    att_updated = 0
    batch = db.batch()
    pending = 0

    async for doc in db.collection("attachments").stream():
        att_scanned += 1
        data = doc.to_dict() or {}
        updates = {}

        # Purge transient URL fields
        for field in ("url", "url_expires_at", "urlExpiresAt"):
            if field in data:
                updates[field] = DELETE_FIELD

        # Backfill bucket and object_name
        bucket = data.get("bucket")
        object_name = data.get("object_name")
        storage_uri = data.get("storage_uri")

        if (not bucket or not object_name) and storage_uri:
            derived_bucket, derived_object = _parse_storage_identity(storage_uri)
            if derived_bucket and not bucket:
                updates["bucket"] = derived_bucket
            if derived_object and not object_name:
                updates["object_name"] = derived_object

        if updates:
            batch.update(doc.reference, updates)
            pending += 1
            att_updated += 1
            if pending >= _BATCH_SIZE:
                await batch.commit()
                batch = db.batch()
                pending = 0

    if pending:
        await batch.commit()
        batch = db.batch()
        pending = 0

    print(
        f"  [001_backfill_storage_identity_and_clean_urls] ✓ Updated {att_updated} attachments ({att_scanned} scanned)."
    )

    # ── 2. Migrate message parts in sessions ───────────────────────────────────
    print("  [001_backfill_storage_identity_and_clean_urls] Scanning session messages for file parts...")
    msg_scanned = 0
    msg_updated = 0

    async for session_doc in db.collection("sessions").stream():
        messages_ref = session_doc.reference.collection("messages")
        async for msg_doc in messages_ref.stream():
            msg_scanned += 1
            msg_data = msg_doc.to_dict() or {}
            parts = msg_data.get("parts") or []
            if not parts:
                continue

            parts_changed = False
            cleaned_parts = []
            for p in parts:
                if isinstance(p, dict) and p.get("type") == "file":
                    clean_p, changed = _clean_file_part(p)
                    cleaned_parts.append(clean_p)
                    if changed:
                        parts_changed = True
                else:
                    cleaned_parts.append(p)

            if parts_changed:
                batch.update(msg_doc.reference, {"parts": cleaned_parts})
                pending += 1
                msg_updated += 1
                if pending >= _BATCH_SIZE:
                    await batch.commit()
                    batch = db.batch()
                    pending = 0

    if pending:
        await batch.commit()
        batch = db.batch()
        pending = 0

    print(
        f"  [001_backfill_storage_identity_and_clean_urls] ✓ Cleaned {msg_updated} session messages "
        f"({msg_scanned} scanned)."
    )

    # ── 3. Migrate shared_chats collection ─────────────────────────────────────
    print("  [001_backfill_storage_identity_and_clean_urls] Scanning shared_chats collection...")
    share_scanned = 0
    share_updated = 0

    async for share_doc in db.collection("shared_chats").stream():
        share_scanned += 1
        share_data = share_doc.to_dict() or {}
        messages = share_data.get("messages") or []
        share_changed = False
        new_messages = []

        for msg in messages:
            if not isinstance(msg, dict):
                new_messages.append(msg)
                continue
            parts = msg.get("parts") or []
            parts_changed = False
            cleaned_parts = []
            for p in parts:
                if isinstance(p, dict) and p.get("type") == "file":
                    clean_p, changed = _clean_file_part(p)
                    cleaned_parts.append(clean_p)
                    if changed:
                        parts_changed = True
                else:
                    cleaned_parts.append(p)

            new_msg = dict(msg)
            if parts_changed:
                new_msg["parts"] = cleaned_parts
                share_changed = True
            new_messages.append(new_msg)

        if share_changed:
            batch.update(share_doc.reference, {"messages": new_messages})
            pending += 1
            share_updated += 1
            if pending >= _BATCH_SIZE:
                await batch.commit()
                batch = db.batch()
                pending = 0

    if pending:
        await batch.commit()

    print(
        f"  [001_backfill_storage_identity_and_clean_urls] ✓ Cleaned {share_updated} shared chats "
        f"({share_scanned} scanned)."
    )

    # ── 4. Cache Invalidation ─────────────────────────────────────────────────
    try:
        from app.core.redis import redis_cache

        await redis_cache.connect()
        await redis_cache.delete_by_prefix("attachments:")
        await redis_cache.delete_by_prefix("sessions:")
        await redis_cache.delete_by_prefix("session_details:")
        await redis_cache.disconnect()
        print("  [001_backfill_storage_identity_and_clean_urls] ✓ Invalidated Redis attachment and session caches")
    except Exception as exc:
        print(f"  [001_backfill_storage_identity_and_clean_urls] (Redis cache invalidation skipped: {exc})")


if __name__ == "__main__":
    asyncio.run(migrate())
