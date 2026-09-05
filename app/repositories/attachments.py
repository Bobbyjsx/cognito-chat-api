import logging
from collections.abc import Sequence
from datetime import datetime
from uuid import UUID

from google.cloud.firestore_v1.async_client import AsyncClient
from google.cloud.firestore_v1.base_query import FieldFilter

from app.models.attachments import AttachmentMetadata

logger = logging.getLogger(__name__)

_IN_QUERY_CHUNK = 30


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
        """Fetch owned attachments by id, chunking Firestore ``in`` queries."""
        if not ids:
            return []
        found: dict[str, AttachmentMetadata] = {}
        unique_ids = list(dict.fromkeys(str(i) for i in ids))
        for start in range(0, len(unique_ids), _IN_QUERY_CHUNK):
            chunk = unique_ids[start : start + _IN_QUERY_CHUNK]
            query = self.collection.where(filter=FieldFilter("id", "in", chunk)).stream()
            async for doc in query:
                data = doc.to_dict() or {}
                if not data or data.get("user_id") != str(user_id):
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

        query = self.collection.where(filter=FieldFilter("user_id", "==", str(user_id)))
        query = query.where(filter=FieldFilter("is_temporary", "==", False))
        if session_id is not None:
            query = query.where(filter=FieldFilter("session_id", "==", str(session_id)))

        # Do not use offset/limit here if we are filtering in python to avoid losing matches
        query = query.order_by("uploaded_at", direction=firestore.Query.DESCENDING)

        results: list[AttachmentMetadata] = []
        async for doc in query.stream():
            data = doc.to_dict() or {}
            if data and data.get("deleted_at") is None:
                meta = AttachmentMetadata(**data)
                results.append(meta)

        # Python-side filtering
        if type:
            if type == "image":
                results = [r for r in results if r.mime_type.startswith("image/")]
            elif type == "document":
                results = [
                    r for r in results if r.mime_type.startswith("application/pdf") or r.mime_type.startswith("text/")
                ]
            else:
                results = [r for r in results if r.type == type]

        if query_string:
            q = query_string.lower()
            results = [r for r in results if q in r.filename.lower()]

        total = len(results)

        # Paginate results
        paginated_results = results[offset : offset + limit]
        has_more = offset + limit < total

        return paginated_results, has_more, total

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

    async def update_storage_location(
        self, attachment_id: UUID | str, storage_uri: str, object_name: str
    ) -> None:
        doc_ref = self.collection.document(str(attachment_id))
        await doc_ref.update({"storage_uri": storage_uri, "object_name": object_name})
