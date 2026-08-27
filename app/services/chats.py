"""Chat agent service.

Coordinates the full chat lifecycle by composing dedicated components —
provider, tool registry/executor, attachment service, and context manager —
instead of growing into one monolithic class.

Provider-agnostic by design: no google-genai imports here. Everything the
model needs (contents, tools, config) is expressed through
``app.providers.base`` types; the provider translates them.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime, timezone

from fastapi import HTTPException

from app.ai.router.blacklist import blacklist_model
from app.models.chats import ChatResponse
from app.models.config import AppConfigDB, normalize_reasoning_level
from app.models.users import UserDB
from app.providers.base import (
    BaseProvider,
    ContentPart,
    GenerationConfig,
    GenerationEvent,
    classify_provider_error,
)
from app.repositories.chats import ChatRepository
from app.repositories.config import ConfigRepository
from app.repositories.users import UserRepository
from app.services.attachments import AttachmentService
from app.services.context import ContextManager
from app.services.quota import resolve_user_limits
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry
from app.utils.datetime import ensure_utc
from app.utils.prompts import get_base_system_instructions

logger = logging.getLogger(__name__)

_REASONING_BUDGETS = {
    "fast": 0,
    "balanced": 8_192,
    "extended": 24_576,
    # Backwards compatibility fallbacks
    "none": 0,
    "minimal": 512,
    "low": 2_048,
    "medium": 8_192,
    "high": 24_576,
    "speed": 0,
    "quality": 24_576,
}


class AgentService:
    def __init__(
        self,
        chat_repo: ChatRepository,
        user_repo: UserRepository,
        config_repo: ConfigRepository,
        attachment_service: AttachmentService,
        provider: BaseProvider | None = None,
        registry: ToolRegistry | None = None,
        executor: ToolExecutor | None = None,
        context_manager: ContextManager | None = None,
        router=None,
    ):
        self.chat_repo = chat_repo
        self.user_repo = user_repo
        self.config_repo = config_repo
        self.attachment_service = attachment_service

        if provider is None:
            from app.core.config import settings
            from app.providers.gemini import GeminiProvider

            provider = GeminiProvider(api_key=settings.gemini_api_key)
        self.provider = provider

        if registry is None:
            registry = ToolRegistry()
            registry.register_defaults()
        self.registry = registry

        self.executor = executor or ToolExecutor(self.registry, self.provider)
        self.context_manager = context_manager or ContextManager()

        if router is None:
            from app.ai.router import SmartModelRouter

            router = SmartModelRouter()
        self.router = router

    # ── configuration ─────────────────────────────────────────────────────────

    async def get_active_config(self) -> AppConfigDB:
        return await self.config_repo.get_config()

    async def validate_and_resolve_config(
        self,
        requested_model: str | None = None,
        requested_reasoning: str | None = None,
        message_text: str | None = None,
        routing_mode: str | None = None,
        context=None,
    ) -> tuple[str, str, int | None, list[dict], list[str]]:
        """Validate model/reasoning against runtime config, invoking smart routing

        if no specific model is requested, and resolve the thinking budget, enabled tools,
        and fallback model candidates.
        """
        cfg = await self.get_active_config()

        if not cfg.enable_text_generation:
            raise HTTPException(status_code=403, detail="Text generation is currently disabled by admin.")

        is_explicit = bool(requested_model and requested_model.lower() not in ("auto", "smart", "default", "none"))
        decision = None
        fallbacks: list[str] = []
        effective_routing_mode = routing_mode
        effective_reasoning = requested_reasoning

        # When smart routing is used, normalize any policy/effort passed
        if not is_explicit and requested_reasoning:
            norm_policy = normalize_reasoning_level(requested_reasoning)
            if norm_policy:
                effective_routing_mode = effective_routing_mode or norm_policy.value
                effective_reasoning = None

        if is_explicit:
            model = requested_model
            if model not in cfg.allowed_text_models:
                raise HTTPException(
                    status_code=400,
                    detail=f"Model '{model}' is not in the allowed text models list: {cfg.allowed_text_models}",
                )
            fallbacks = [m for m in cfg.allowed_text_models if m != model]
        elif cfg.enable_smart_routing and self.router and message_text:
            model, fallbacks, decision = await self.router.route_or_default(
                message=message_text,
                context=context,
                requested_model=requested_model,
                policy=effective_routing_mode,
                config=cfg,
            )
        else:
            model = cfg.default_text_model
            fallbacks = [m for m in cfg.allowed_text_models if m != model]

        allowed_for_model = cfg.get_reasoning_modes_for_model(model)
        allowed_vals = [m.value if hasattr(m, "value") else str(m) for m in allowed_for_model]
        global_allowed_vals = [m.value if hasattr(m, "value") else str(m) for m in cfg.allowed_reasoning_levels]

        # Resolve reasoning mode
        if effective_reasoning:
            norm_reasoning = normalize_reasoning_level(effective_reasoning)
            req_str = norm_reasoning.value if norm_reasoning else str(effective_reasoning).lower().strip()

            if req_str not in global_allowed_vals and str(effective_reasoning) not in global_allowed_vals:
                raise HTTPException(
                    status_code=400,
                    detail=f"Reasoning level '{effective_reasoning}' is not allowed in global system config. Allowed: {global_allowed_vals}",
                )

            if req_str in allowed_vals:
                reasoning = req_str
            elif is_explicit:
                raise HTTPException(
                    status_code=400,
                    detail=f"Reasoning level '{effective_reasoning}' is not allowed for model '{model}'. Allowed: {allowed_vals}",
                )
            elif decision and decision.analysis:
                analysis = decision.analysis
                if (analysis.reasoning_required >= 0.70 or analysis.complexity >= 0.75) and "extended" in allowed_vals:
                    reasoning = "extended"
                elif (
                    analysis.reasoning_required >= 0.35 or analysis.complexity >= 0.45
                ) and "balanced" in allowed_vals:
                    reasoning = "balanced"
                elif "fast" in allowed_vals:
                    reasoning = "fast"
                else:
                    reasoning = allowed_vals[0] if allowed_vals else "fast"
            else:
                reasoning = allowed_vals[0] if allowed_vals else "fast"
        elif decision and decision.analysis:
            # Dynamically right-size reasoning based on request complexity and reasoning signals
            analysis = decision.analysis
            if (analysis.reasoning_required >= 0.70 or analysis.complexity >= 0.75) and "extended" in allowed_vals:
                reasoning = "extended"
            elif (analysis.reasoning_required >= 0.35 or analysis.complexity >= 0.45) and "balanced" in allowed_vals:
                reasoning = "balanced"
            elif "fast" in allowed_vals:
                reasoning = "fast"
            else:
                reasoning = allowed_vals[0] if allowed_vals else "fast"
        else:
            default_val = (
                cfg.default_reasoning_level.value
                if hasattr(cfg.default_reasoning_level, "value")
                else str(cfg.default_reasoning_level)
            )
            reasoning = default_val if default_val in allowed_vals else (allowed_vals[0] if allowed_vals else "fast")

        thinking_budget = _REASONING_BUDGETS.get(reasoning, 0)

        # Smart tool filtering: only attach heavy external tools (code_exec / google_search) when explicitly required
        if decision and decision.analysis:
            analysis = decision.analysis
            from app.models.config import ToolName

            if analysis.coding_required >= 0.4 or analysis.task_type.value == "coding":
                allowed_tools = [t for t in cfg.allowed_tools if t != ToolName.GOOGLE_SEARCH]
            elif analysis.web_required:
                allowed_tools = [t for t in cfg.allowed_tools if t != ToolName.CODE_EXECUTION]
            else:
                allowed_tools = [
                    t for t in cfg.allowed_tools if t not in (ToolName.CODE_EXECUTION, ToolName.GOOGLE_SEARCH)
                ]
            tool_configs = self.registry.to_provider_configs(allowed_tools)
        else:
            from app.models.config import ToolName

            allowed_tools = list(cfg.allowed_tools)
            if ToolName.CODE_EXECUTION in allowed_tools and ToolName.GOOGLE_SEARCH in allowed_tools:
                allowed_tools = [t for t in allowed_tools if t != ToolName.CODE_EXECUTION]
            tool_configs = self.registry.to_provider_configs(allowed_tools)

        logger.info(
            "[AgentService] Resolved model config: model='%s', reasoning='%s', thinking_budget=%s, tools=%s, fallbacks=%s",
            model,
            reasoning,
            thinking_budget,
            [c.get("kind") or c.get("name") for c in tool_configs],
            fallbacks,
        )

        return model, reasoning, thinking_budget, tool_configs, fallbacks

    # ── quota ─────────────────────────────────────────────────────────────────

    async def _quota_precheck(self, user: UserDB, config: AppConfigDB) -> None:
        now = datetime.now(timezone.utc)
        reset_at = ensure_utc(user.reset_at)
        weekly_reset_at = ensure_utc(user.weekly_reset_at)

        is_6h_expired = reset_at is None or reset_at <= now
        is_weekly_expired = weekly_reset_at is None or weekly_reset_at <= now

        effective_6h = 0 if is_6h_expired else user.tokens_used_6h
        effective_weekly = 0 if is_weekly_expired else user.tokens_used_weekly

        limit_6h, limit_weekly = resolve_user_limits(user, config)

        if effective_6h >= limit_6h:
            reset_str = reset_at.isoformat() if reset_at else ""
            raise HTTPException(status_code=429, detail=f"6-hour token limit reached. Resets at {reset_str}")
        if effective_weekly >= limit_weekly:
            reset_str = weekly_reset_at.isoformat() if weekly_reset_at else ""
            raise HTTPException(status_code=429, detail=f"Weekly token limit reached. Resets at {reset_str}")

    async def _charge_usage(self, user: UserDB, tokens: int, config: AppConfigDB) -> bool:
        if tokens <= 0:
            return True
        success = await self.user_repo.atomic_increment_if_within_limit(
            user.id,
            tokens,
            default_limit_6h=config.default_token_limit_6h,
            default_limit_weekly=config.default_token_limit_weekly,
        )
        if success:
            from app.core.cache_keys import CacheKeys
            from app.core.redis import redis_cache

            await redis_cache.delete(CacheKeys.user_profile(user.id))
            await redis_cache.delete(CacheKeys.user_auth(user.id))
        return success

    # ── session handling ──────────────────────────────────────────────────────

    @staticmethod
    async def _generate_title(first_message: str) -> str:
        """Instantly derives a concise title from the user prompt without network latency."""
        words = first_message.strip().split()
        return " ".join(words[:5]) if words else "New Chat"

    async def _resolve_session(
        self,
        user: UserDB,
        session_id: uuid.UUID | None,
        message_text: str,
    ) -> tuple[object, uuid.UUID, str | None]:
        title = None
        if session_id is None:
            title = await self._generate_title(message_text)
            session = await self.chat_repo.create_session(user_id=user.id, title=title)
            session_id = session.id
        else:
            session, _ = await self.chat_repo.get_session(session_id, user_id=user.id)
            if not session:
                raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
            if not session.title:
                title = await self._generate_title(message_text)
                await self.chat_repo.update_session_title(session_id, title)
            else:
                title = session.title
        return session, session_id, title

    # ── attachments ───────────────────────────────────────────────────────────

    async def _prepare_attachments(
        self,
        user: UserDB,
        config: AppConfigDB,
        attachment_ids: list[uuid.UUID] | None,
        session_id: uuid.UUID | None,
    ) -> list:
        if not attachment_ids:
            return []
        if len(attachment_ids) > config.attachment_max_count:
            raise HTTPException(
                status_code=400,
                detail=f"Too many attachments. Maximum is {config.attachment_max_count} per message.",
            )
        if not config.enable_attachments:
            raise HTTPException(status_code=403, detail="Attachments are currently disabled by admin.")

        attachments = await self.attachment_service.resolve_many(user.id, attachment_ids)
        missing = {str(i) for i in attachment_ids} - {str(a.id) for a in attachments}
        if missing:
            raise HTTPException(status_code=400, detail=f"Attachment(s) not found: {sorted(missing)}")

        for attachment in attachments:
            if attachment.session_id is None and session_id is not None:
                await self.attachment_service.bind_session(attachment, session_id)

        if attachment_ids:
            await self.attachment_service.make_permanent(user.id, attachment_ids)
            attachments = await self.attachment_service.resolve_many(user.id, attachment_ids)

        return attachments

    async def _prepare_current_parts(
        self,
        message_text: str,
        attachments: list,
    ) -> list[dict]:
        parts = [{"text": message_text}]
        for attachment in attachments:
            parts.extend(await self.attachment_service.prepare_parts(attachment))
        return parts

    # ── history / context ─────────────────────────────────────────────────────

    async def _build_contents(
        self,
        user: UserDB,
        session: object | None,
        config: AppConfigDB,
        current_parts: list[dict],
    ) -> list[ContentPart]:
        history = session.messages if session is not None and session.messages else []
        if config.context_trim_enabled:
            history = self.context_manager.trim(
                history,
                max_tokens=config.context_max_tokens,
                keep_recent=config.context_keep_recent,
            )

        attachment_ids = [a_id for msg in history for a_id in (msg.attachment_ids or [])]
        attachment_map: dict[str, object] = {}
        if attachment_ids:
            try:
                parsed = [uuid.UUID(str(i)) for i in attachment_ids]
            except ValueError:
                parsed = []
            for meta in await self.attachment_service.resolve_many(user.id, parsed):
                attachment_map[str(meta.id)] = meta

        contents: list[ContentPart] = []
        for msg in history:
            parts: list[dict] = [{"text": msg.content}]
            for a_id in msg.attachment_ids or []:
                meta = attachment_map.get(str(a_id))
                if meta is None:
                    continue
                try:
                    parts.extend(await self.attachment_service.prepare_parts(meta))
                except Exception:
                    logger.exception("Failed to prepare historical attachment %s", a_id)
            contents.append(ContentPart(role="user" if msg.role == "user" else "model", parts=parts))

        contents.append(ContentPart(role="user", parts=current_parts))
        return contents

    # ── SSE helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _serialize_event(event: GenerationEvent) -> dict:
        if event.type in ("text", "reasoning"):
            return {"type": event.type, "token": event.token}
        if event.type == "tool_call" and event.tool_call is not None:
            call = event.tool_call
            return {
                "type": "tool_call",
                "tool_name": call.name,
                "tool_call_id": call.id,
                "input": call.args,
            }
        if event.type == "tool_result" and event.tool_result is not None:
            result = event.tool_result
            return {
                "type": "tool_result",
                "tool_name": result.name,
                "tool_call_id": result.id,
                "output": result.output,
            }
        if event.type == "usage":
            return {"type": "usage", "tokens": event.total_tokens}
        return {"type": event.type, "token": event.token}

    @staticmethod
    def _error_event(detail: str, code: str | None = None) -> str:
        payload = {"detail": detail}
        if code:
            payload["code"] = code
        return f"event: error\ndata: {json.dumps(payload)}\n\n"

    # ── non-streaming ─────────────────────────────────────────────────────────

    async def process_chat(
        self,
        user: UserDB,
        message_text: str,
        session_id: uuid.UUID | None = None,
        requested_model: str | None = None,
        requested_reasoning: str | None = None,
        attachment_ids: list[uuid.UUID] | None = None,
        routing_mode: str | None = None,
    ) -> ChatResponse:
        if not message_text or not message_text.strip():
            raise HTTPException(status_code=400, detail="Message is empty.")

        active_config = await self.get_active_config()
        await self._quota_precheck(user, active_config)

        session, session_id, title = await self._resolve_session(user, session_id, message_text)
        attachments = await self._prepare_attachments(user, active_config, attachment_ids, session_id)

        from app.ai.router.schemas import RequestContext
        from app.models.chats import MessageRole

        routing_context = RequestContext(
            conversation_message_count=len(session.messages) if session and session.messages else 0,
            approximate_context_tokens=len(message_text) // 4,
            has_attachments=bool(attachments),
            attachment_types=[a.type for a in attachments],
            user_id=str(user.id),
            session_id=str(session_id),
        )

        model, reasoning, thinking_budget, tool_configs, fallbacks = await self.validate_and_resolve_config(
            requested_model=requested_model,
            requested_reasoning=requested_reasoning,
            message_text=message_text,
            routing_mode=routing_mode,
            context=routing_context,
        )

        await self.chat_repo.add_message(
            session_id,
            role=MessageRole.USER,
            content=message_text,
            attachment_ids=[str(a.id) for a in attachments],
        )

        current_parts = await self._prepare_current_parts(message_text, attachments)
        contents = await self._build_contents(user, session, active_config, current_parts)
        generation_config = GenerationConfig(
            system_instruction=get_base_system_instructions(),
            thinking_budget=thinking_budget,
            include_thoughts=True,
            tool_configs=tool_configs,
        )

        # Attempt primary model with fallback candidates if available
        models_to_attempt = [model] + [m for m in fallbacks if m != model]
        result = None
        used_model = model
        last_exc = None

        for attempt_idx, candidate_model in enumerate(models_to_attempt):
            try:
                result = await self.executor.generate(candidate_model, contents, generation_config)
                used_model = candidate_model
                break
            except Exception as exc:
                last_exc = exc
                await blacklist_model(candidate_model)
                if attempt_idx < len(models_to_attempt) - 1:
                    logger.warning(
                        "Generation with model '%s' failed (%s). Attempting fallback '%s'.",
                        candidate_model,
                        exc,
                        models_to_attempt[attempt_idx + 1],
                    )
                else:
                    logger.exception(
                        "Model generation failed session_id=%s model=%s",
                        session_id,
                        candidate_model,
                    )

        if result is None and last_exc is not None:
            status, error_code, message = classify_provider_error(last_exc)
            await self.chat_repo.add_message(
                session_id,
                role=MessageRole.AGENT,
                content="",
                error=f"[{error_code}] {message}",
            )
            raise HTTPException(
                status_code=status,
                detail={"code": error_code, "message": message},
            ) from last_exc

        await self.chat_repo.add_message(session_id, role=MessageRole.AGENT, content=result.text)

        within_limit = await self._charge_usage(user, result.total_tokens, active_config)
        if not within_limit:
            raise HTTPException(
                status_code=429, detail="Token limit exceeded after generation. Please upgrade your plan."
            )

        from app.core.redis import redis_cache

        await redis_cache.delete_by_prefix(f"sessions:{user.id}")
        await redis_cache.delete_by_prefix(f"session:{session_id}")

        logger.info(
            "[AgentService] Chat completed: session_id=%s, model='%s', tokens=%d, response='%s...'",
            session_id,
            used_model,
            result.total_tokens,
            result.text[:80].replace("\n", " ") if result.text else "",
        )

        return ChatResponse(
            session_id=session_id,
            title=title,
            response=result.text,
            model=used_model,
            reasoning=reasoning,
        )

    # ── streaming ─────────────────────────────────────────────────────────────

    async def stream_chat(
        self,
        user: UserDB,
        message_text: str,
        session_id: uuid.UUID | None = None,
        requested_model: str | None = None,
        requested_reasoning: str | None = None,
        attachment_ids: list[uuid.UUID] | None = None,
        routing_mode: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """Yields SSE-formatted chunks as the provider responds, then persists
        the full message and token usage once the stream is complete."""
        if not message_text or not message_text.strip():
            yield self._error_event("Message is empty.")
            return

        import asyncio

        try:
            # 1. Concurrently fetch active config and resolve chat session
            active_config, (session, session_id, title) = await asyncio.gather(
                self.get_active_config(),
                self._resolve_session(user, session_id, message_text),
            )
            await self._quota_precheck(user, active_config)

            # Flush session event immediately to establish SSE connection and stream headers (< 10ms)
            yield f"event: session\ndata: {json.dumps({'session_id': str(session_id), 'title': title})}\n\n"

            attachments = await self._prepare_attachments(user, active_config, attachment_ids, session_id)

            from app.ai.router.schemas import RequestContext
            from app.models.chats import MessageRole

            routing_context = RequestContext(
                conversation_message_count=len(session.messages) if session and session.messages else 0,
                approximate_context_tokens=len(message_text) // 4,
                has_attachments=bool(attachments),
                attachment_types=[a.type for a in attachments],
                user_id=str(user.id),
                session_id=str(session_id),
            )

            model, reasoning, thinking_budget, tool_configs, fallbacks = await self.validate_and_resolve_config(
                requested_model=requested_model,
                requested_reasoning=requested_reasoning,
                message_text=message_text,
                routing_mode=routing_mode,
                context=routing_context,
            )

            # 2. Persist user message in the background concurrently without blocking stream start
            user_msg_task = asyncio.create_task(
                self.chat_repo.add_message(
                    session_id,
                    role=MessageRole.USER,
                    content=message_text,
                    attachment_ids=[str(a.id) for a in attachments],
                )
            )

            current_parts = await self._prepare_current_parts(message_text, attachments)
            contents = await self._build_contents(user, session, active_config, current_parts)
            generation_config = GenerationConfig(
                system_instruction=get_base_system_instructions(),
                thinking_budget=thinking_budget,
                include_thoughts=True,
                tool_configs=tool_configs,
            )
        except HTTPException as exc:
            yield self._error_event(str(exc.detail))
            return
        except Exception as exc:
            logger.exception("[AgentService] Stream setup failed for session_id=%s", session_id)
            yield self._error_event(f"Internal error during stream setup: {exc}")
            return

        models_to_attempt = [model] + [m for m in fallbacks if m != model]
        used_model = model
        last_exc = None
        has_yielded_chunks = False
        full_response = ""
        total_tokens = 0

        for attempt_idx, candidate_model in enumerate(models_to_attempt):
            full_response = ""
            total_tokens = 0
            has_yielded_chunks = False
            last_exc = None
            try:
                logger.info(
                    "[AgentService] Invoking provider generate_stream for model='%s' with %d contents",
                    candidate_model,
                    len(contents),
                )
                async for event in self.executor.generate_stream(candidate_model, contents, generation_config):
                    if event.type == "text":
                        full_response += event.token or ""
                    elif event.type == "usage" and event.total_tokens:
                        total_tokens = event.total_tokens
                    yield f"event: chunk\ndata: {json.dumps(self._serialize_event(event))}\n\n"
                    has_yielded_chunks = True

                used_model = candidate_model
                break
            except Exception as exc:
                last_exc = exc
                await blacklist_model(candidate_model)
                if has_yielded_chunks:
                    logger.warning("Stream failed mid-generation for model '%s', cannot retry.", candidate_model)
                    break
                else:
                    if attempt_idx < len(models_to_attempt) - 1:
                        logger.warning(
                            "Generation stream with model '%s' failed (%s). Attempting fallback '%s'.",
                            candidate_model,
                            exc,
                            models_to_attempt[attempt_idx + 1],
                        )
                    else:
                        logger.exception(
                            "Model generation stream failed session_id=%s model=%s",
                            session_id,
                            candidate_model,
                        )

        if last_exc is not None:
            status, error_code, message = classify_provider_error(last_exc)
            logger.exception(
                "Model generation failed session_id=%s model=%s error_code=%s status=%s",
                session_id,
                used_model,
                error_code,
                status,
            )
            await asyncio.gather(
                user_msg_task,
                self.chat_repo.add_message(
                    session_id,
                    role=MessageRole.AGENT,
                    content=full_response,
                    error=f"[{error_code}] {message}",
                ),
                return_exceptions=True,
            )
            yield self._error_event(message, error_code)
            return

        from app.core.redis import redis_cache

        # Concurrently ensure user message is persisted, agent response is stored, quota is charged, and cache is evicted
        persist_results = await asyncio.gather(
            user_msg_task,
            self.chat_repo.add_message(session_id, role=MessageRole.AGENT, content=full_response),
            self._charge_usage(user, total_tokens, active_config),
            redis_cache.delete_by_prefix(f"sessions:{user.id}"),
            redis_cache.delete_by_prefix(f"session:{session_id}"),
            return_exceptions=True,
        )

        within_limit = persist_results[2] if isinstance(persist_results[2], bool) else True
        if not within_limit:
            yield self._error_event("Token limit exceeded after generation. Please upgrade your plan.")
            return

        logger.info(
            "[AgentService] Stream chat completed: session_id=%s, model='%s', tokens=%d, full_response='%s...'",
            session_id,
            used_model,
            total_tokens,
            full_response[:80].replace("\n", " ") if full_response else "",
        )

        yield (
            "event: done\n"
            f"data: {json.dumps({'session_id': str(session_id), 'title': title, 'tokens_used': total_tokens, 'model': used_model, 'reasoning': reasoning})}\n\n"
        )
