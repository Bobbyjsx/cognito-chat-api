import logging
from collections.abc import Sequence
from datetime import datetime
from uuid import UUID

from google.cloud.firestore_v1.async_client import AsyncClient
from google.cloud.firestore_v1.base_query import FieldFilter

from app.models.attachments import AttachmentMetadata

logger = logging.getLogger(__name__)

_GET_ALL_CHUNK = 100
_LIST_DELETED_SLACK = 8


class AttachmentRepository:
    def __init__(self, db: AsyncClient):
        self.db = db
        self.collection = self.db.collection("attachments")

    async def create(self, metadata: AttachmentMetadata) -> AttachmentMetadata:
        doc_ref = self.collection.document(str(metadata.id))
        await doc_ref.set(metadata.model_dump(mode="json"))
        return metadata

    async def get(
        self, attachment_id: UUID | str, user_id: UUID | str, include_deleted: bool = False
    ) -> AttachmentMetadata | None:
        doc_ref = self.collection.document(str(attachment_id))
        doc = await doc_ref.get()
        if not doc.exists:
            return None
        data = doc.to_dict() or {}
        if not data or data.get("user_id") != str(user_id):
            return None
        if not include_deleted and data.get("deleted_at") is not None:
            return None
        return AttachmentMetadata(**data)

    async def get_many(
        self, user_id: UUID | str, ids: Sequence[UUID | str], include_deleted: bool = False
    ) -> list[AttachmentMetadata]:
        """Fetch owned attachments by document id via batched ``get_all``."""
        if not ids:
            return []
        found: dict[str, AttachmentMetadata] = {}
        unique_ids = list(dict.fromkeys(str(i) for i in ids))
        owner = str(user_id)
        for start in range(0, len(unique_ids), _GET_ALL_CHUNK):
            chunk = unique_ids[start : start + _GET_ALL_CHUNK]
            refs = [self.collection.document(doc_id) for doc_id in chunk]
            async for doc in self.db.get_all(refs):
                if not doc.exists:
                    continue
                data = doc.to_dict() or {}
                if not data or data.get("user_id") != owner:
                    continue
                if not include_deleted and data.get("deleted_at") is not None:
                    continue
                meta = AttachmentMetadata(**data)
                found[str(meta.id)] = meta
        return [found[i] for i in unique_ids if i in found]

    async def list_by_user(
        self,
        user_id: UUID | str,
        session_id: UUID | str | None = None,
        type: str | None = None,
        query_string: str | None = None,
        limit: int = 15,
        offset: int = 0,
    ) -> tuple[list[AttachmentMetadata], bool, int]:
        from google.cloud import firestore

        limit = max(1, limit)
        offset = max(0, offset)
        query = self.collection.where(filter=FieldFilter("user_id", "==", str(user_id)))
        query = query.where(filter=FieldFilter("is_temporary", "==", False))
        if session_id is not None:
            query = query.where(filter=FieldFilter("session_id", "==", str(session_id)))

        query = query.order_by("uploaded_at", direction=firestore.Query.DESCENDING)
        filename_q = query_string.lower() if query_string else None
        python_type = type
        if not python_type and not filename_q:
            query = query.limit(offset + limit + 1 + _LIST_DELETED_SLACK)

        results: list[AttachmentMetadata] = []
        skipped = 0
        async for doc in query.stream():
            data = doc.to_dict() or {}
            if not data or data.get("deleted_at") is not None:
                continue
            meta = AttachmentMetadata(**data)
            if python_type:
                if python_type == "image" and not meta.mime_type.startswith("image/"):
                    continue
                if python_type == "document" and not (
                    meta.mime_type.startswith("application/pdf") or meta.mime_type.startswith("text/")
                ):
                    continue
                if python_type not in ("image", "document") and meta.type != python_type:
                    continue
            if filename_q and filename_q not in meta.filename.lower():
                continue
            if skipped < offset:
                skipped += 1
                continue
            results.append(meta)
            if len(results) > limit:
                break

        has_more = len(results) > limit
        if has_more:
            results = results[:limit]
        total = offset + len(results) + (1 if has_more else 0)
        return results, has_more, total

    async def list_abandoned_temporary(self, before: datetime) -> list[AttachmentMetadata]:
        query = self.collection.where(filter=FieldFilter("is_temporary", "==", True))
        query = query.where(filter=FieldFilter("uploaded_at", "<", before))

        results: list[AttachmentMetadata] = []
        async for doc in query.stream():
            data = doc.to_dict() or {}
            if data:
                meta = AttachmentMetadata(**data)
                results.append(meta)
        return results

    async def update_session(self, attachment_id: UUID | str, session_id: UUID | str) -> None:
        doc_ref = self.collection.document(str(attachment_id))
        await doc_ref.update({"session_id": str(session_id)})

    async def update_gemini_uri(self, attachment_id: UUID | str, gemini_file_uri: str | None) -> None:
        doc_ref = self.collection.document(str(attachment_id))
        await doc_ref.update({"gemini_file_uri": gemini_file_uri})

    async def delete(self, attachment_id: UUID | str) -> None:
        doc_ref = self.collection.document(str(attachment_id))
        await doc_ref.delete()

    async def soft_delete(self, attachment_id: UUID | str) -> None:
        from datetime import datetime, timezone

        doc_ref = self.collection.document(str(attachment_id))
        now = datetime.now(timezone.utc)
        await doc_ref.update({"deleted_at": now.isoformat()})

    async def update_temporary_flag(self, attachment_id: UUID | str, is_temporary: bool) -> None:
        doc_ref = self.collection.document(str(attachment_id))
        await doc_ref.update({"is_temporary": is_temporary})

    async def update_storage_uri(self, attachment_id: UUID | str, storage_uri: str) -> None:
        doc_ref = self.collection.document(str(attachment_id))
        await doc_ref.update({"storage_uri": storage_uri})

    async def update_storage_location(self, attachment_id: UUID | str, storage_uri: str, object_name: str) -> None:
        doc_ref = self.collection.document(str(attachment_id))
        await doc_ref.update({"storage_uri": storage_uri, "object_name": object_name})
