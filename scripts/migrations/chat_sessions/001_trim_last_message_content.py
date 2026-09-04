import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from app.database import create_db_client, init_db
from app.models.chats import SESSION_LIST_PREVIEW_CHARS, clip_session_preview

_BATCH_SIZE = 400


async def migrate():
    """Trim denormalized session.last_message_content to the list preview length.

    Date: 2026-09-04
    Idempotent: Only writes when the stored preview is longer than SESSION_LIST_PREVIEW_CHARS.
    Full message bodies in the messages subcollection are left unchanged.
    """
    print("  [001_trim_last_message_content] Initializing Firestore connection...")
    init_db()
    db = create_db_client()

    scanned = 0
    updated = 0
    batch = db.batch()
    pending = 0

    async for doc in db.collection("sessions").stream():
        scanned += 1
        data = doc.to_dict() or {}
        content = data.get("last_message_content")
        clipped = clip_session_preview(content)
        if clipped == content:
            continue

        batch.update(doc.reference, {"last_message_content": clipped})
        pending += 1
        updated += 1
        if pending >= _BATCH_SIZE:
            await batch.commit()
            print(f"  [001_trim_last_message_content] Committed {updated} updates ({scanned} scanned)")
            batch = db.batch()
            pending = 0

    if pending:
        await batch.commit()

    print(
        f"  [001_trim_last_message_content] ✓ Trimmed {updated} session previews "
        f"(scanned {scanned}, max {SESSION_LIST_PREVIEW_CHARS} chars)."
    )

    try:
        from app.core.redis import redis_cache

        await redis_cache.connect()
        await redis_cache.delete_by_prefix("sessions:")
        await redis_cache.disconnect()
        print("  [001_trim_last_message_content] ✓ Invalidated Redis session list caches")
    except Exception as exc:
        print(f"  [001_trim_last_message_content] (Redis cache invalidation skipped: {exc})")


if __name__ == "__main__":
    asyncio.run(migrate())
