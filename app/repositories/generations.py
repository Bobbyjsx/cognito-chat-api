import asyncio
import contextlib
import logging
import time
from datetime import datetime, timezone
from uuid import UUID

from google.cloud.firestore_v1.async_client import AsyncClient
from google.cloud.firestore_v1.base_query import FieldFilter

from app.core.config import settings
from app.models.chats import GenerationDB, GenerationStatus

logger = logging.getLogger(__name__)

# Maximum allowed duration (in seconds) for a generation before auto-marking as failed
GENERATION_TIMEOUT_SECONDS = settings.generation_timeout_seconds
_ACTIVE_GENS_TTL_SECONDS = 1.5
_active_gens_memo: dict[str, tuple[float, dict[str, str]]] = {}


class GenerationRepository:
    def __init__(self, db: AsyncClient):
        self.db = db
        self.collection = self.db.collection("generations")

    async def get_active_generations_for_user(self, user_id: UUID | str) -> dict[str, str]:
        memo_key = str(user_id)
        hit = _active_gens_memo.get(memo_key)
        now_mono = time.monotonic()
        if hit and now_mono - hit[0] < _ACTIVE_GENS_TTL_SECONDS:
            return dict(hit[1])

        docs = (
            self.collection.where(filter=FieldFilter("user_id", "==", str(user_id)))
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
            .limit(50)
            .stream()
        )
        mapping = {}
        now = datetime.now(timezone.utc)
        async for doc in docs:
            data = doc.to_dict() or {}
            if not data:
                continue
            session_id = data.get("session_id")
            # Auto-expire generations with no heartbeat/update > GENERATION_TIMEOUT_SECONDS ago
            check_time_val = data.get("updated_at") or data.get("created_at")
            if check_time_val:
                with contextlib.suppress(Exception):
                    if isinstance(check_time_val, str):
                        check_time = datetime.fromisoformat(check_time_val)
                    else:
                        check_time = check_time_val
                    if check_time.tzinfo is None:
                        check_time = check_time.replace(tzinfo=timezone.utc)
                    if (now - check_time).total_seconds() > GENERATION_TIMEOUT_SECONDS:
                        asyncio.create_task(
                            self.update_status(
                                doc.id,
                                GenerationStatus.FAILED,
                                error=f"Generation timed out after {GENERATION_TIMEOUT_SECONDS} seconds",
                            )
                        )
                        continue
            if session_id:
                mapping[str(session_id)] = str(doc.id)
        if len(_active_gens_memo) > 1024:
            _active_gens_memo.clear()
        _active_gens_memo[memo_key] = (now_mono, mapping)
        return mapping

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
        data = docs[0].to_dict() or {}
        if not data:
            return None
        generation = GenerationDB(**data)
        now = datetime.now(timezone.utc)
        check_time = generation.updated_at or generation.created_at
        if check_time.tzinfo is None:
            check_time = check_time.replace(tzinfo=timezone.utc)
        if (now - check_time).total_seconds() > GENERATION_TIMEOUT_SECONDS:
            await self.update_status(
                generation.id,
                GenerationStatus.FAILED,
                error=f"Generation timed out after {GENERATION_TIMEOUT_SECONDS} seconds",
            )

    async def cancel_active_generations_for_session(
        self, session_id: UUID | str, reason: str = "Session deleted by user"
    ) -> int:
        """Finds and cancels all non-terminal generations for a session."""
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
            .get()
        )
        count = 0
        for doc in docs:
            await self.update_status(
                doc.id,
                GenerationStatus.CANCELLED,
                error=reason,
            )
            count += 1
        return count

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
        data = doc.to_dict() or {}
        if not data:
            return None
        generation = GenerationDB(**data)

        # Check timeout expiration for active/non-terminal generations
        if generation.status in {
            GenerationStatus.QUEUED,
            GenerationStatus.RUNNING_LIVE,
            GenerationStatus.RUNNING_WORKER,
        }:
            now = datetime.now(timezone.utc)
            check_time = generation.updated_at or generation.created_at
            if check_time.tzinfo is None:
                check_time = check_time.replace(tzinfo=timezone.utc)
            elapsed = (now - check_time).total_seconds()
            if elapsed > GENERATION_TIMEOUT_SECONDS:
                logger.warning(
                    "Generation %s timed out after %.1fs (timeout limit: %ds). Marking as FAILED.",
                    generation.id,
                    elapsed,
                    GENERATION_TIMEOUT_SECONDS,
                )
                await self.update_status(
                    generation.id,
                    GenerationStatus.FAILED,
                    error=f"Generation timed out after {int(elapsed)} seconds",
                )
                generation.status = GenerationStatus.FAILED
                generation.error = f"Generation timed out after {int(elapsed)} seconds"

        return generation

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

        # When generation fails, attach the error directly to the corresponding user prompt message.
        if status == GenerationStatus.FAILED:
            try:
                doc = await doc_ref.get()
                if doc.exists:
                    gen_data = doc.to_dict() or {}
                    session_id = gen_data.get("session_id")
                    user_id = gen_data.get("user_id")
                    user_message_id = gen_data.get("user_message_id") or gen_data.get("message_id")
                    error_msg = (
                        kwargs.get("error") or gen_data.get("error") or "Model generation failed. Please try again."
                    )

                    if session_id:
                        from google.cloud import firestore

                        from app.core.cache_keys import CacheKeys
                        from app.core.redis import redis_cache
                        from app.repositories.chats import ChatRepository

                        chat_repo = ChatRepository(self.db)
                        if user_message_id:
                            await chat_repo.update_message(session_id, user_message_id, error=error_msg)
                        else:
                            # Fallback: update the latest user message in this session
                            last_user_msgs = (
                                await self.db.collection("sessions")
                                .document(str(session_id))
                                .collection("messages")
                                .where(filter=FieldFilter("role", "==", "user"))
                                .order_by("created_at", direction=firestore.Query.DESCENDING)
                                .limit(1)
                                .get()
                            )
                            if last_user_msgs:
                                await last_user_msgs[0].reference.update({"error": error_msg})

                        if user_id:
                            await redis_cache.delete_by_prefix(CacheKeys.user_sessions_prefix(user_id))
                        await redis_cache.delete_by_prefix(CacheKeys.session_details_prefix(session_id))
            except Exception as e:
                logger.error(
                    "Failed to update error on message for generation %s: %s",
                    generation_id,
                    e,
                )

    async def update(self, generation_id: UUID | str, **fields) -> None:
        """Update arbitrary fields on a generation document."""
        doc_ref = self.collection.document(str(generation_id))
        update_data = {k: str(v) if isinstance(v, UUID) else v for k, v in fields.items()}
        update_data["updated_at"] = datetime.now(timezone.utc).isoformat()
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

        res = await _transition_in_transaction(self.db.transaction())
        return bool(res)
