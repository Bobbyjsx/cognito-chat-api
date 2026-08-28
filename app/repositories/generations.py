import logging
from datetime import datetime, timezone
from uuid import UUID

from google.cloud.firestore_v1.async_client import AsyncClient
from google.cloud.firestore_v1.base_query import FieldFilter

from app.models.chats import GenerationDB, GenerationStatus

logger = logging.getLogger(__name__)


class GenerationRepository:
    def __init__(self, db: AsyncClient):
        self.db = db
        self.collection = self.db.collection("generations")

    async def get_active_generation(self, session_id: UUID | str) -> GenerationDB | None:
        docs = (
            await self.collection.where(filter=FieldFilter("session_id", "==", str(session_id)))
            .where(
                filter=FieldFilter(
                    "status",
                    "in",
                    [
                        GenerationStatus.QUEUED.value,
                        GenerationStatus.RUNNING_LIVE.value,
                        GenerationStatus.RUNNING_WORKER.value,
                    ],
                )
            )
            .limit(1)
            .get()
        )
        if not docs:
            return None
        return GenerationDB(**docs[0].to_dict())

    async def create(self, generation: GenerationDB) -> GenerationDB:
        doc_ref = self.collection.document(str(generation.id))
        data = generation.model_dump(mode="json")
        await doc_ref.set(data)
        return generation

    async def get_by_id(self, generation_id: UUID | str) -> GenerationDB | None:
        doc_ref = self.collection.document(str(generation_id))
        doc = await doc_ref.get()
        if not doc.exists:
            return None
        return GenerationDB(**doc.to_dict())

    async def update_status(self, generation_id: UUID | str, status: GenerationStatus, **kwargs) -> None:
        doc_ref = self.collection.document(str(generation_id))
        update_data = {
            "status": status.value,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if "error" in kwargs:
            update_data["error"] = kwargs["error"]
        if "completed_at" in kwargs:
            update_data["completed_at"] = kwargs["completed_at"]
        if "usage_tokens" in kwargs:
            update_data["usage_tokens"] = kwargs["usage_tokens"]
        if "buffered_text" in kwargs:
            update_data["buffered_text"] = kwargs["buffered_text"]
        if "buffered_thoughts" in kwargs:
            update_data["buffered_thoughts"] = kwargs["buffered_thoughts"]
        if "message_id" in kwargs:
            update_data["message_id"] = str(kwargs["message_id"])

        await doc_ref.update(update_data)

    async def heartbeat(
        self,
        generation_id: UUID | str,
        buffered_text: str | None = None,
        buffered_thoughts: str | None = None,
    ) -> None:
        """Update the updated_at timestamp to indicate the generation is still alive."""
        doc_ref = self.collection.document(str(generation_id))
        update_data = {"updated_at": datetime.now(timezone.utc).isoformat()}
        if buffered_text is not None:
            update_data["buffered_text"] = buffered_text
        if buffered_thoughts is not None:
            update_data["buffered_thoughts"] = buffered_thoughts
        try:
            await doc_ref.update(update_data)
        except Exception as e:
            logger.warning(f"Failed to heartbeat generation {generation_id}: {e}")

    async def atomic_transition_status(
        self,
        generation_id: UUID | str,
        target_status: GenerationStatus,
        expected_current_statuses: list[GenerationStatus] | None = None,
    ) -> bool:
        """Atomically claim/transition a generation."""
        doc_ref = self.collection.document(str(generation_id))

        from google.cloud.firestore_v1.async_transaction import async_transactional

        @async_transactional
        async def _transition_in_transaction(transaction) -> bool:
            doc = await doc_ref.get(transaction=transaction)
            if not doc.exists:
                return False

            current_status = doc.get("status")
            expected_values = [s.value for s in expected_current_statuses] if expected_current_statuses else None

            if expected_values and current_status not in expected_values:
                return False

            transaction.update(
                doc_ref,
                {
                    "status": target_status.value,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            return True

        return await _transition_in_transaction(self.db.transaction())
