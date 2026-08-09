import logging
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

    async def get(self, attachment_id: UUID, user_id: UUID) -> AttachmentMetadata | None:
        doc_ref = self.collection.document(str(attachment_id))
        doc = await doc_ref.get()
        if not doc.exists:
            return None
        data = doc.to_dict()
        if data.get("user_id") != str(user_id):
            return None
        return AttachmentMetadata(**data)

    async def get_many(self, user_id: UUID, ids: list[UUID]) -> list[AttachmentMetadata]:
        """Fetch owned attachments by id, chunking Firestore ``in`` queries."""
        if not ids:
            return []
        found: dict[str, AttachmentMetadata] = {}
        unique_ids = list(dict.fromkeys(str(i) for i in ids))
        for start in range(0, len(unique_ids), _IN_QUERY_CHUNK):
            chunk = unique_ids[start : start + _IN_QUERY_CHUNK]
            query = self.collection.where(filter=FieldFilter("id", "in", chunk)).stream()
            async for doc in query:
                data = doc.to_dict()
                if data.get("user_id") != str(user_id):
                    continue
                meta = AttachmentMetadata(**data)
                found[str(meta.id)] = meta
        return [found[i] for i in unique_ids if i in found]

    async def list_by_user(self, user_id: UUID, session_id: UUID | None = None) -> list[AttachmentMetadata]:
        query = self.collection.where(filter=FieldFilter("user_id", "==", str(user_id)))
        if session_id is not None:
            query = query.where(filter=FieldFilter("session_id", "==", str(session_id)))
            
        results: list[AttachmentMetadata] = []
        async for doc in query.stream():
            data = doc.to_dict()
            meta = AttachmentMetadata(**data)
            results.append(meta)
        results.sort(key=lambda m: m.uploaded_at, reverse=True)
        return results

    async def list_abandoned_temporary(self, before: datetime) -> list[AttachmentMetadata]:
        query = self.collection.where(filter=FieldFilter("is_temporary", "==", True))
        query = query.where(filter=FieldFilter("uploaded_at", "<", before))
        
        results: list[AttachmentMetadata] = []
        async for doc in query.stream():
            data = doc.to_dict()
            meta = AttachmentMetadata(**data)
            results.append(meta)
        return results

    async def update_session(self, attachment_id: UUID, session_id: UUID) -> None:
        doc_ref = self.collection.document(str(attachment_id))
        await doc_ref.update({"session_id": str(session_id)})

    async def update_gemini_uri(self, attachment_id: UUID, gemini_file_uri: str) -> None:
        doc_ref = self.collection.document(str(attachment_id))
        await doc_ref.update({"gemini_file_uri": gemini_file_uri})

    async def delete(self, attachment_id: UUID) -> None:
        doc_ref = self.collection.document(str(attachment_id))
        await doc_ref.delete()

    async def update_temporary_flag(self, attachment_id: UUID, is_temporary: bool) -> None:
        doc_ref = self.collection.document(str(attachment_id))
        await doc_ref.update({"is_temporary": is_temporary})

    async def update_storage_uri(self, attachment_id: UUID, storage_uri: str) -> None:
        doc_ref = self.collection.document(str(attachment_id))
        await doc_ref.update({"storage_uri": storage_uri})
