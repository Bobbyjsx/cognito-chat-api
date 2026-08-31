import asyncio
import logging
import time
from datetime import datetime, timezone

from app.models.chats import GenerationStatus, MessageRole
from app.repositories.chats import ChatRepository
from app.repositories.config import ConfigRepository
from app.repositories.generations import GenerationRepository
from app.repositories.users import UserRepository
from app.schemas.task import GenerationTaskPayload, TaskExecutionResponse
from app.services.chats import AgentService

logger = logging.getLogger(__name__)


class GenerationWorkerService:
    def __init__(
        self,
        generation_repo: GenerationRepository,
        chat_repo: ChatRepository,
        user_repo: UserRepository,
        config_repo: ConfigRepository,
        agent_service: AgentService,
    ) -> None:
        self.generation_repo = generation_repo
        self.chat_repo = chat_repo
        self.user_repo = user_repo
        self.config_repo = config_repo
        self.agent_service = agent_service

    async def execute_task(self, task: GenerationTaskPayload) -> TaskExecutionResponse:
        generation_id = task.generation_id
        attempt_num = task.attempt_number
        start_time = time.perf_counter()

        logger.info(
            "Worker starting generation task %s (attempt %d)",
            generation_id,
            attempt_num,
        )

        generation = await self.generation_repo.get_by_id(generation_id)
        if not generation:
            logger.error("Generation %s not found in Firestore", generation_id)
            return TaskExecutionResponse(
                status="failed_not_found",
                generation_id=generation_id,
                attempt_number=attempt_num,
                duration_ms=(time.perf_counter() - start_time) * 1000,
            )

        if generation.status in {
            GenerationStatus.COMPLETED,
            GenerationStatus.FAILED,
            GenerationStatus.CANCELLED,
        }:
            logger.info("Generation %s already in terminal state '%s'", generation_id, generation.status.value)
            return TaskExecutionResponse(
                status="already_processed",
                generation_id=generation_id,
                attempt_number=attempt_num,
                duration_ms=(time.perf_counter() - start_time) * 1000,
            )

        if generation.status == GenerationStatus.RUNNING_LIVE:
            # Check heartbeat
            now = datetime.now(timezone.utc)
            updated_at = generation.updated_at
            if (now - updated_at).total_seconds() < 15.0:
                logger.info("Generation %s is actively running live (recent heartbeat). Yielding.", generation_id)
                # Raising a transient error will cause Cloud Tasks to retry later
                raise ValueError("Generation is actively running live.")

        # Claim the generation
        claimed = await self.generation_repo.atomic_transition_status(
            generation_id,
            GenerationStatus.RUNNING_WORKER,
            expected_current_statuses=[
                GenerationStatus.QUEUED,
                GenerationStatus.RUNNING_LIVE,
                GenerationStatus.RUNNING_WORKER,
            ],
        )

        if not claimed:
            logger.info("Failed to claim generation %s, maybe already completed or claimed.", generation_id)
            return TaskExecutionResponse(
                status="claim_failed",
                generation_id=generation_id,
                attempt_number=attempt_num,
                duration_ms=(time.perf_counter() - start_time) * 1000,
            )

        user = await self.user_repo.get_by_id(str(generation.user_id))
        if not user:
            logger.error("User %s not found for generation %s", generation.user_id, generation_id)
            return TaskExecutionResponse(
                status="failed_no_user",
                generation_id=generation_id,
                attempt_number=attempt_num,
                duration_ms=(time.perf_counter() - start_time) * 1000,
            )

        # Reconstruct the generation contents and execute
        # Wait, the easiest way is to let AgentService do it, but AgentService takes message_text.
        # But the user's message is already in Firestore.
        # We need to construct the history up to the user message, and run generation.
        # Let's extract that logic or re-implement here briefly.

        try:
            active_config = await self.agent_service.get_active_config()
            session, _ = await self.chat_repo.get_session(generation.session_id, user.id, limit=100)
            if not session:
                raise ValueError("Session not found")

            # The last message from the user should be the trigger message.
            # But what if there are newer messages? We should ideally only consider messages up to when generation was created.
            # For simplicity (V1), we just take all messages, the last user message being the prompt.
            # If the last message is an agent message (e.g. from another generation), we might need to be careful.

            contents = []
            for msg in session.messages:
                if not msg.content and msg.role == MessageRole.AGENT:
                    continue
                parts = [{"text": msg.content}]
                from app.providers.base import ContentPart

                contents.append(ContentPart(role="user" if msg.role == MessageRole.USER else "model", parts=parts))

            # If there's no message in the session, something is wrong
            if not contents:
                raise ValueError("No messages found in session.")

            from app.providers.base import GenerationConfig
            from app.services.chats import get_base_system_instructions

            model = generation.resolved_model
            reasoning = generation.resolved_reasoning
            fallbacks: list[str] = []

            if not model:
                last_user_text = ""
                for msg in reversed(session.messages):
                    if msg.role == MessageRole.USER:
                        last_user_text = msg.content
                        break

                (
                    resolved_model,
                    resolved_reasoning,
                    thinking_budget,
                    tool_configs,
                    fallbacks,
                ) = await self.agent_service.validate_and_resolve_config(
                    requested_model=generation.requested_model,
                    requested_reasoning=generation.requested_reasoning,
                    message_text=last_user_text,
                )
                model = resolved_model
                reasoning = resolved_reasoning
            else:
                thinking_budget = 0
                if reasoning == "extended":
                    thinking_budget = 24576
                elif reasoning == "balanced":
                    thinking_budget = 8192

                tool_configs = self.agent_service.registry.to_provider_configs(
                    [t.value for t in active_config.allowed_tools]
                )

            generation_config = GenerationConfig(
                system_instruction=get_base_system_instructions(),
                thinking_budget=thinking_budget,
                include_thoughts=True,
                tool_configs=tool_configs,
            )

            models_to_attempt = [model] + [m for m in fallbacks if m != model]
            full_response = ""
            full_thoughts = ""
            total_tokens = 0

            for attempt_idx, candidate_model in enumerate(models_to_attempt):
                full_response = ""
                full_thoughts = ""
                total_tokens = 0
                last_heartbeat_time = asyncio.get_running_loop().time()
                try:
                    logger.info(
                        "Worker executing generate_stream for model='%s' with %d contents",
                        candidate_model,
                        len(contents),
                    )
                    async for event in self.agent_service.provider.generate_stream(
                        candidate_model, contents, generation_config
                    ):
                        if event.type == "text":
                            full_response += event.token or ""
                        elif event.type == "reasoning":
                            full_thoughts += event.token or ""
                        elif event.type == "usage" and event.total_tokens:
                            total_tokens = event.total_tokens

                        now = asyncio.get_running_loop().time()
                        if now - last_heartbeat_time > 2.0:
                            asyncio.create_task(
                                self.generation_repo.heartbeat(
                                    generation.id, buffered_text=full_response, buffered_thoughts=full_thoughts
                                )
                            )
                            last_heartbeat_time = now

                    logger.info("Worker completed streaming successfully with model '%s'", candidate_model)
                    break
                except Exception as exc:
                    logger.warning("Worker attempt for model '%s' failed: %s", candidate_model, exc)
                    if attempt_idx >= len(models_to_attempt) - 1:
                        raise

            # Record success
            agent_msg = await self.chat_repo.add_message(
                session.id, role=MessageRole.AGENT, content=full_response, generation_id=str(generation.id)
            )

            await self.agent_service._charge_usage(user, total_tokens, active_config)
            await self.generation_repo.update_status(
                generation_id,
                GenerationStatus.COMPLETED,
                completed_at=datetime.now(timezone.utc).isoformat(),
                usage_tokens=total_tokens,
                buffered_text=full_response,
                buffered_thoughts=full_thoughts,
                message_id=str(agent_msg.id),
            )

            from app.core.redis import redis_cache
            from app.core.cache_keys import CacheKeys

            await redis_cache.delete_by_prefix(CacheKeys.user_sessions_prefix(user.id))
            await redis_cache.delete_by_prefix(CacheKeys.session_details_prefix(session.id))

            logger.info("Worker successfully completed generation %s", generation_id)
            return TaskExecutionResponse(
                status="sent",
                generation_id=generation_id,
                attempt_number=attempt_num,
                duration_ms=(time.perf_counter() - start_time) * 1000,
            )

        except Exception as exc:
            logger.error("Worker failed generation %s: %s", generation_id, exc)
            await self.generation_repo.update_status(generation_id, GenerationStatus.FAILED, error=str(exc))
            return TaskExecutionResponse(
                status="failed_permanent",
                generation_id=generation_id,
                attempt_number=attempt_num,
                duration_ms=(time.perf_counter() - start_time) * 1000,
            )
