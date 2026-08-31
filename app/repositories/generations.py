import asyncio
import logging
from datetime import datetime, timezone
from uuid import UUID

from google.cloud.firestore_v1.async_client import AsyncClient
from google.cloud.firestore_v1.base_query import FieldFilter

from app.models.chats import GenerationDB, GenerationStatus

logger = logging.getLogger(__name__)

# Maximum allowed duration (in seconds) for a generation before auto-marking as failed
GENERATION_TIMEOUT_SECONDS = 30


class GenerationRepository:
    def __init__(self, db: AsyncClient):
        self.db = db
        self.collection = self.db.collection("generations")

    async def get_active_generations_for_user(self, user_id: UUID | str) -> dict[str, str]:
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
            .stream()
        )
        mapping = {}
        now = datetime.now(timezone.utc)
        async for doc in docs:
            data = doc.to_dict()
            session_id = data.get("session_id")
            # Auto-expire generations with no heartbeat/update > GENERATION_TIMEOUT_SECONDS ago
            check_time_val = data.get("updated_at") or data.get("created_at")
            if check_time_val:
                try:
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
                except Exception:
                    pass
            if session_id:
                mapping[str(session_id)] = str(doc.id)
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
        generation = GenerationDB(**docs[0].to_dict())
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
            return None
        return generation

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
        generation = GenerationDB(**doc.to_dict())

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
        from datetime import timedelta

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

        # When generation fails, ensure an agent failure message exists in the session's message list,
        # bound to the generation's historical creation time rather than 'now'.
        if status == GenerationStatus.FAILED:
            try:
                doc = await doc_ref.get()
                if doc.exists:
                    gen_data = doc.to_dict()
                    session_id = gen_data.get("session_id")
                    user_id = gen_data.get("user_id")
                    error_msg = (
                        kwargs.get("error") or gen_data.get("error") or "Model generation failed. Please try again."
                    )

                    if session_id:
                        from app.models.chats import MessageRole
                        from app.repositories.chats import ChatRepository

                        # Check if an agent message for this generation already exists
                        existing_msgs = (
                            await self.db.collection("sessions")
                            .document(str(session_id))
                            .collection("messages")
                            .where(filter=FieldFilter("generation_id", "==", str(generation_id)))
                            .where(filter=FieldFilter("role", "in", [MessageRole.AGENT.value, "agent", "model"]))
                            .limit(1)
                            .get()
                        )
                        if not existing_msgs:
                            gen_created_at = gen_data.get("created_at")
                            if gen_created_at:
                                if isinstance(gen_created_at, str):
                                    msg_time = datetime.fromisoformat(gen_created_at)
                                else:
                                    msg_time = gen_created_at
                                if msg_time.tzinfo is None:
                                    msg_time = msg_time.replace(tzinfo=timezone.utc)
                                msg_time = msg_time + timedelta(milliseconds=1)
                            else:
                                msg_time = datetime.now(timezone.utc)

                            # Check if newer messages already exist in this session
                            newer_msgs = (
                                await self.db.collection("sessions")
                                .document(str(session_id))
                                .collection("messages")
                                .where(filter=FieldFilter("created_at", ">", msg_time.isoformat()))
                                .limit(1)
                                .get()
                            )
                            should_update_summary = len(newer_msgs) == 0

                            chat_repo = ChatRepository(self.db)
                            await chat_repo.add_message(
                                session_id=UUID(str(session_id)),
                                role=MessageRole.AGENT,
                                content="",
                                error=error_msg,
                                generation_id=str(generation_id),
                                created_at=msg_time,
                                update_session_summary=should_update_summary,
                            )
                            # Invalidate redis cache for session details and user sessions
                            from app.core.cache_keys import CacheKeys
                            from app.core.redis import redis_cache

                            if user_id:
                                await redis_cache.delete_by_prefix(CacheKeys.user_sessions_prefix(user_id))
                            await redis_cache.delete_by_prefix(CacheKeys.session_details_prefix(session_id))
            except Exception as e:
                logger.error(
                    "Failed to append failure message for generation %s: %s",
                    generation_id,
                    e,
                )

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
