"""Chat agent service.

Coordinates the full chat lifecycle by composing dedicated components —
provider, tool registry/executor, attachment service, and context manager —
instead of growing into one monolithic class.

Provider-agnostic by design: no google-genai imports here. Everything the
model needs (contents, tools, config) is expressed through
``app.providers.base`` types; the provider translates them.
"""

from __future__ import annotations

import asyncio

_fire_and_forget_tasks = set()

import json
import logging
import re
import time
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from fastapi import HTTPException, Request

from app.ai.router.blacklist import blacklist_model
from app.models.attachments import AttachmentMetadata
from app.models.chats import ChatResponse, ChatSessionDB, MessageRole
from app.models.config import AppConfigDB, normalize_reasoning_level
from app.models.users import UserDB
from app.providers.base import (
    BaseProvider,
    ContentPart,
    GenerationConfig,
    GenerationEvent,
    classify_provider_error,
    is_retryable_provider_error,
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
        provider_registry: Any | None = None,
        registry: ToolRegistry | None = None,
        executor: ToolExecutor | None = None,
        context_manager: ContextManager | None = None,
        router=None,
        generation_repo=None,
        tasks_dispatcher=None,
    ):
        self.chat_repo = chat_repo
        self.user_repo = user_repo
        self.config_repo = config_repo
        self.attachment_service = attachment_service
        self.db = getattr(chat_repo, "db", None)

        if generation_repo is None and self.db is not None:
            from app.repositories.generations import GenerationRepository

            generation_repo = GenerationRepository(self.db)
        self.generation_repo = generation_repo
        self.tasks_dispatcher = tasks_dispatcher

        if provider_registry is None:
            if provider is not None:
                from app.models.config import ModelProvider
                from app.providers.registry import ProviderRegistry

                provider_registry = ProviderRegistry()
                provider_registry.register("default", provider)
                provider_registry.register("gemini", provider)
                provider_registry.register("google", provider)
                provider_registry.register(ModelProvider.GOOGLE, provider)
                provider_registry.register("anthropic", provider)
                provider_registry.register("claude", provider)
                provider_registry.register(ModelProvider.ANTHROPIC, provider)
            else:
                from app.providers.registry import create_default_provider_registry

                provider_registry = create_default_provider_registry()

        self.provider_registry = provider_registry
        self.provider = provider or provider_registry.get("gemini")

        if registry is None:
            registry = ToolRegistry()
            registry.register_defaults()
        self.registry = registry

        self.executor = executor or ToolExecutor(self.registry, self.provider_registry)
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

        # When model is 'auto' (not explicit), use reasoning as the routing policy.
        # When model is explicit, reasoning acts as the model reasoning effort.
        # This unifies both into `reasoning`, making `routing_mode` redundant.
        if not is_explicit and requested_reasoning:
            norm_policy = normalize_reasoning_level(requested_reasoning)
            if norm_policy:
                effective_routing_mode = effective_routing_mode or norm_policy.value

        if is_explicit:
            model = requested_model
            if model not in cfg.allowed_text_models:
                raise HTTPException(
                    status_code=400,
                    detail=f"Model '{model}' is not in the allowed text models list: {cfg.allowed_text_models}",
                )
            fallbacks = [m for m in cfg.allowed_text_models if m != model and m.lower() != "auto"]
        elif cfg.enable_smart_routing and self.router and message_text:
            model, fallbacks, decision = await self.router.route_or_default(
                message=message_text,
                context=context,
                requested_model=requested_model,
                policy=effective_routing_mode,
                config=cfg,
            )
            fallbacks = [m for m in fallbacks if m != model and m.lower() != "auto"]
        else:
            model = cfg.default_text_model
            fallbacks = [m for m in cfg.allowed_text_models if m != model and m.lower() != "auto"]

        # Ensure model is never virtual 'auto'
        if model.lower() == "auto":
            model = cfg.default_text_model if cfg.default_text_model.lower() != "auto" else "gemini-3.5-flash"

        allowed_for_model = cfg.get_reasoning_modes_for_model(model)
        allowed_vals = [m.value if isinstance(m, Enum) else str(m) for m in allowed_for_model]
        global_allowed_vals = [m.value if isinstance(m, Enum) else str(m) for m in cfg.allowed_reasoning_levels]

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

        # Smart tool filtering: dynamically attach tools (code_exec / google_search) based on request requirements
        analysis = decision.analysis if decision else None
        if analysis is None and message_text and self.router:
            try:
                from app.ai.router.analyzer import BaseRequestAnalyzer

                analyzer = getattr(self.router, "analyzer", None)
                heuristic = getattr(analyzer, "fallback_analyzer", None) or getattr(
                    self.router, "heuristic_analyzer", None
                )
                if isinstance(heuristic, BaseRequestAnalyzer):
                    analysis = await heuristic.analyze(message_text, context=context)
                elif analyzer and hasattr(analyzer, "analyze"):
                    analysis = await analyzer.analyze(message_text, context=context)
            except Exception as exc:
                logger.warning("[AgentService] Dynamic request analysis for tools failed: %s", exc)

        model_cfg = cfg.models_list.get(model)
        supports_search = model_cfg.supports_web_search if model_cfg else True
        supports_code = model_cfg.supports_code_execution if model_cfg else True

        from app.models.config import ToolName

        if analysis:
            if (analysis.coding_required >= 0.4 or analysis.task_type.value == "coding") and supports_code:
                allowed_tools = [t for t in cfg.allowed_tools if t == ToolName.CODE_EXECUTION]
            elif (
                analysis.web_required or analysis.task_type.value in ("general_knowledge", "information")
            ) and supports_search:
                allowed_tools = [t for t in cfg.allowed_tools if t == ToolName.GOOGLE_SEARCH]
            else:
                allowed_tools = [
                    t for t in cfg.allowed_tools if t not in (ToolName.CODE_EXECUTION, ToolName.GOOGLE_SEARCH)
                ]
            tool_configs = self.registry.to_provider_configs([t.value for t in allowed_tools])
        else:
            allowed_tools = list(cfg.allowed_tools)
            if not supports_code:
                allowed_tools = [t for t in allowed_tools if t != ToolName.CODE_EXECUTION]
            if not supports_search:
                allowed_tools = [t for t in allowed_tools if t != ToolName.GOOGLE_SEARCH]
            tool_configs = self.registry.to_provider_configs([t.value for t in allowed_tools])

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
        from app.core.cache_keys import CacheKeys
        from app.core.redis import redis_cache

        try:
            await redis_cache.delete(CacheKeys.user_profile(user.id))
            await redis_cache.delete(CacheKeys.user_auth(user.id))
        except Exception as exc:
            logger.debug("Redis cache invalidation skipped for user %s: %s", user.id, exc)
        return success

    # ── session handling ──────────────────────────────────────────────────────

    @staticmethod
    def sanitize_title_prompt(text: str) -> str:
        """Extract clean text for title generation, removing code blocks, urls, and limiting chars to reduce token cost."""
        if not text or not text.strip():
            return ""
        cleaned = re.sub(r"```[\s\S]*?```", "", text)
        cleaned = re.sub(r"`[^`]*`", "", cleaned)
        cleaned = re.sub(r"https?://\S+", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        # Strictly limit characters to minimize cost (no attachments, short prompt)
        return cleaned[:160].strip()

    @classmethod
    def _evaluate_title_strategy(cls, first_message: str) -> tuple[str, bool]:
        """
        Dynamically chooses between local and AI title generation based on prompt structure.
        Returns: (initial_title, needs_ai_worker)

        If the prompt matches a high-confidence template pattern, returns (summary, False)
        so it resolves locally at 0 cost and 0 latency.
        If no high-confidence rule matches, returns (fast_fallback_title, True) so that
        the session has an immediate title and AI title generation is offloaded to the worker.
        """
        if not first_message or not first_message.strip():
            return "New Chat", False

        text = re.sub(r"```[\s\S]*?```", "", first_message)
        text = re.sub(r"`[^`]*`", "", text)
        text = re.sub(r"https?://\S+", "", text)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = lines[0] if lines else first_message.strip()

        patterns = [
            (
                r"^(?:(?:i\s+(?:need|want)\s+)?help\s+(?:me\s+)?(?:with|do|finish|on)\s+(?:my\s+)?|can\s+you\s+(?:please\s+)?help\s+(?:me\s+)?(?:with|do|finish)\s+(?:my\s+)?)([a-zA-Z0-9\s]+?)\s+(?:assignment|homework|project|essay|paper|coursework|exam|test)",
                r"Assistance with \1 assignment",
            ),
            (
                r"^(?:(?:i\s+(?:need|want)\s+)?help\s+(?:me\s+)?(?:with|on)\s+|can\s+you\s+(?:please\s+)?help\s+(?:me\s+)?(?:with|on)\s+)(.+)",
                r"Help with \1",
            ),
            (
                r"^(?:(?:can|could|would)\s+(?:you\s+)?(?:please\s+)?|please\s+)?(?:write|create|build|code|implement)\s+(?:a|an)?\s*(.+?)\s*(?:script|tool|program|bot|app|api|service)",
                r"\1 script",
            ),
            (
                r"^(?:(?:can|could|would)\s+(?:you\s+)?(?:please\s+)?|please\s+)?(?:explain|tell\s+me\s+about|teach\s+me)\s+(?:the\s+)?(.+)",
                r"Overview of \1",
            ),
            (
                r"^(?:how\s+(?:do\s+i|to|can\s+we|can\s+i)\s+)(.+)",
                r"Guide to \1",
            ),
            (
                r"^(?:debug|fix|troubleshoot|solve)\s+(?:this\s+|my\s+)?(.+)",
                r"Debugging \1",
            ),
            (
                r"^(?:what\s+(?:is|are|was|were)\s+(?:the\s+)?)(.+)",
                r"Guide to \1",
            ),
            (
                r"^(?:why\s+(?:is|are|does|do)\s+)(.+)",
                r"Analysis of \1",
            ),
            (
                r"^(?:summarize|review)\s+(?:the\s+)?(.+)",
                r"Summary of \1",
            ),
        ]

        for pat, repl in patterns:
            m = re.match(pat, text, flags=re.IGNORECASE)
            if m:
                summary = m.expand(repl).strip()
                summary = re.sub(r"[\?\.!\:\;]+$", "", summary).strip()
                summary = re.sub(
                    r"\s+(in simple terms|for beginners|step by step|please)$",
                    "",
                    summary,
                    flags=re.IGNORECASE,
                ).strip()
                words = summary.split()
                if len(words) <= 6:
                    return (summary[0].upper() + summary[1:]), False
                truncated = " ".join(words[:5])
                return (truncated[0].upper() + truncated[1:]), False

        # Greeting check: simple greetings use "Conversation" as immediate title and offload to AI worker
        greeting_match = re.match(
            r"^(hi|hello|hey|greetings|good morning|good afternoon|good evening|yo)\b",
            text,
            flags=re.IGNORECASE,
        )
        if greeting_match and len(text.split()) <= 3:
            return "Conversation", True

        # Generic clean extraction for fast immediate title while AI worker processes
        clean = re.sub(
            r"^(can|could|would) you (please\s+)?(tell me |explain |help me with |show me |write |create |summarize )?",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()
        clean = re.sub(r"[\?\.!\:\;]+$", "", clean).strip()
        clean = re.sub(
            r"\s+(in simple terms|for beginners|step by step|please)$",
            "",
            clean,
            flags=re.IGNORECASE,
        ).strip()
        words = clean.split() or first_message.strip().split()
        summary = " ".join(words[:5]).strip()
        initial_title = (summary[0].upper() + summary[1:]) if summary else "Conversation"
        return initial_title, True

    @staticmethod
    def _is_generic_title(title: str | None) -> bool:
        """Determines if a session title is a generic greeting or placeholder that should be upgraded on substantive messages."""
        if not title:
            return True
        t = title.strip().lower()
        generic_exact = {
            "new chat",
            "conversation",
            "greeting",
            "greetings",
            "hello",
            "hey",
            "casual conversation",
            "general conversation",
            "general conversation initiation",
            "casual greeting and conversation",
            "general greeting and assistance",
        }
        return t in generic_exact or any(
            t.startswith(prefix) for prefix in ("casual greeting", "general conversation", "general greeting")
        )

    @staticmethod
    def _rule_based_title_summary(first_message: str) -> str:
        """Derives a semantic summary title using pattern templates when LLM is unavailable."""
        title, _ = AgentService._evaluate_title_strategy(first_message)
        return title

    async def generate_ai_title(self, prompt: str, model: str | None = None) -> str | None:
        """
        Invokes LLM with minimal prompt snippet (no attachments, max 160 chars) to produce a 3-5 word summary title.
        Runs strictly in background worker to never block chat streaming TTFT.
        """
        clean_snippet = self.sanitize_title_prompt(prompt)
        if not clean_snippet:
            return None

        try:
            cfg = await self.get_active_config()
            target_model = (
                model or getattr(cfg, "title_generation_model", "gemini-3.1-flash-lite") or "gemini-3.1-flash-lite"
            )
            if target_model.lower() == "auto":
                target_model = (
                    getattr(cfg, "title_generation_model", "gemini-3.1-flash-lite") or "gemini-3.1-flash-lite"
                )

            config = GenerationConfig(
                system_instruction=(
                    "You generate short, descriptive chat session titles. "
                    "Summarize the user's intent or topic into 3 to 5 words. "
                    "Examples:\n"
                    "- User: 'Can you help me do my geography assignment about tectonic plates?' -> Title: Assistance with geography assignment\n"
                    "- User: 'Write a python script to scrape prices from amazon' -> Title: Amazon price scraper script\n"
                    "- User: 'Why does my docker container exit with code 137?' -> Title: Debugging Docker exit 137\n"
                    "- User: 'Teach me how to play chess' -> Title: Beginner chess tutorial\n"
                    "Do NOT copy the user prompt verbatim. Summarize their goal/task. "
                    "Do NOT use quotes, punctuation, or conversational filler. Return ONLY the title text."
                ),
                thinking_budget=0,
                include_thoughts=False,
                tool_configs=[],
            )
            content = [
                ContentPart(
                    role="user",
                    parts=[{"text": f"User: {clean_snippet}"}],
                )
            ]
            res = await self.executor.generate(target_model, content, config)
            raw_title = (res.text or "").strip().strip("\"'*`").rstrip(".")
            raw_title = re.sub(r"^(title|summary):\s*", "", raw_title, flags=re.IGNORECASE).strip()
            if raw_title and 3 <= len(raw_title) <= 55 and "\n" not in raw_title:
                return raw_title[0].upper() + raw_title[1:]
        except Exception as exc:
            logger.debug("[AgentService] generate_ai_title failed: %s", exc)

        return None

    async def enqueue_title_generation(
        self,
        session_id: uuid.UUID | str,
        user_id: uuid.UUID | str,
        prompt_snippet: str,
        model: str | None = None,
    ) -> None:
        """Offloads AI title generation to Cloud Tasks worker or local background worker."""
        clean_snippet = self.sanitize_title_prompt(prompt_snippet)
        if not clean_snippet:
            return

        from app.schemas.task import TitleTaskPayload

        payload = TitleTaskPayload(
            session_id=str(session_id),
            user_id=str(user_id),
            prompt=clean_snippet,
            model=model,
        )

        enqueued = False
        if self.tasks_dispatcher is not None:
            try:
                await self.tasks_dispatcher.enqueue_title_task(payload)
                enqueued = True
            except Exception as e:
                logger.warning(
                    "[AgentService] Cloud Tasks title enqueue failed (%s). Falling back to background worker.", e
                )

        if not enqueued:
            try:
                from app.services.generation_worker import GenerationWorkerService

                worker = GenerationWorkerService(
                    generation_repo=self.generation_repo,
                    chat_repo=self.chat_repo,
                    user_repo=self.user_repo,
                    config_repo=self.config_repo,
                    agent_service=self,
                )
                task = asyncio.create_task(worker.execute_title_task(payload))
                _fire_and_forget_tasks.add(task)
                task.add_done_callback(_fire_and_forget_tasks.discard)
            except Exception as e:
                logger.error("[AgentService] Failed to dispatch local title worker task: %s", e)

    async def _resolve_session(
        self,
        user: UserDB,
        session_id: uuid.UUID | None,
        message_text: str,
        model: str | None = None,
    ) -> tuple[ChatSessionDB, uuid.UUID, str, bool]:
        """
        Resolves or creates the session, assigning an initial title to ensure it is stored
        in the DB, and deciding whether AI title generation should be offloaded to the worker.
        Returns: (session, session_id, initial_title, needs_ai_worker)
        """
        initial_title, needs_ai_worker = self._evaluate_title_strategy(message_text)

        if session_id is None:
            session = await self.chat_repo.create_session(user_id=user.id, title=initial_title)
            session_id = session.id
        else:
            session, _ = await self.chat_repo.get_session(session_id, user_id=user.id)
            if not session:
                raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
            if not session.title or self._is_generic_title(session.title):
                await self.chat_repo.update_session_title(session_id, initial_title, user_id=user.id)
                session.title = initial_title
            else:
                initial_title = session.title
                needs_ai_worker = False

        return session, session_id, initial_title, needs_ai_worker

    # ── attachments ───────────────────────────────────────────────────────────

    async def _prepare_attachments(
        self,
        user: UserDB,
        config: AppConfigDB,
        attachment_ids: list[uuid.UUID] | None,
        session_id: uuid.UUID | None,
    ) -> list[AttachmentMetadata]:
        if not attachment_ids:
            return []
        if len(attachment_ids) > config.attachment_max_count:
            raise HTTPException(
                status_code=400,
                detail=f"Too many attachments. Maximum allowed is {config.attachment_max_count}.",
            )
        attachments = await self.attachment_service.resolve_many(user.id, attachment_ids)
        if len(attachments) != len(attachment_ids):
            raise HTTPException(status_code=400, detail="One or more attachments not found or not owned by you.")
        # If this is a new session that was just created, bind the attachments to it
        # and make them permanent (remove from temp)
        if session_id:
            for att in attachments:
                if att.session_id is None:
                    await self.attachment_service.bind_session(att, session_id)
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

    # ── prompt construction ───────────────────────────────────────────────────

    async def _build_contents(
        self,
        user: UserDB,
        session: ChatSessionDB | None,
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
        attachment_map: dict[str, AttachmentMetadata] = {}
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
                "args": call.args,
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
        payload = {"detail": detail, "error": detail}
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

        _session_res = await self._resolve_session(user, session_id, message_text, model=requested_model)
        if len(_session_res) == 4:
            session, session_id, title, needs_ai_worker = _session_res
        else:
            session, session_id, title = _session_res[:3]
            needs_ai_worker = False

        if needs_ai_worker and getattr(active_config, "enable_ai_title_generation", True):
            _enqueue_task = asyncio.create_task(
                self.enqueue_title_generation(session_id, user.id, message_text, model=requested_model)
            )
            _fire_and_forget_tasks.add(_enqueue_task)
            _enqueue_task.add_done_callback(_fire_and_forget_tasks.discard)

        attachments = await self._prepare_attachments(user, active_config, attachment_ids, session_id)

        from app.ai.router.schemas import RequestContext

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

        user_parts = [{"type": "text", "text": message_text}] if message_text else []
        for a in attachments:
            user_parts.append(
                {
                    "type": "file",
                    "attachment_id": str(a.id),
                    "filename": a.filename,
                    "contentType": a.mime_type,
                    "mediaType": a.mime_type,
                    "size": a.size,
                    "bucket": a.bucket,
                    "object_name": a.object_name,
                    "storage_uri": a.storage_uri,
                }
            )

        await self.chat_repo.add_message(
            session_id,
            role=MessageRole.USER,
            content=message_text,
            attachment_ids=[str(a.id) for a in attachments],
            parts=user_parts,
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
        models_to_attempt = [m for m in [model] + [m for m in fallbacks if m != model] if m and m.lower() != "auto"]
        result = None
        used_model = model
        last_exc = None
        gen_start_time = time.perf_counter()

        for attempt_idx, candidate_model in enumerate(models_to_attempt):
            try:
                result = await self.executor.generate(candidate_model, contents, generation_config)
                used_model = candidate_model
                break
            except Exception as exc:
                last_exc = exc
                await blacklist_model(candidate_model)
                is_retryable = is_retryable_provider_error(exc)

                if attempt_idx < len(models_to_attempt) - 1 and is_retryable:
                    logger.warning(
                        "Generation with model '%s' failed with retryable error (%s). Attempting fallback '%s'.",
                        candidate_model,
                        exc,
                        models_to_attempt[attempt_idx + 1],
                    )
                else:
                    if not is_retryable:
                        logger.warning(
                            "Generation with model '%s' failed with non-retryable error (%s). Bypassing fallbacks.",
                            candidate_model,
                            exc,
                        )
                    else:
                        logger.exception(
                            "Model generation failed session_id=%s model=%s",
                            session_id,
                            candidate_model,
                        )
                    break

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

        assert result is not None
        proc_parts: list[dict[str, Any]] = []
        if getattr(result, "tool_calls", None):
            for call in result.tool_calls:
                proc_parts.append(
                    {
                        "type": "dynamic-tool",
                        "toolName": call.name,
                        "tool_name": call.name,
                        "toolCallId": call.id,
                        "tool_call_id": call.id,
                        "state": "output-available",
                        "input": call.args or {},
                        "output": {},
                    }
                )
        if result.text:
            proc_parts.append({"type": "text", "text": result.text})
        await self.chat_repo.add_message(session_id, role=MessageRole.AGENT, content=result.text, parts=proc_parts)

        within_limit = await self._charge_usage(user, result.total_tokens, active_config)
        if not within_limit:
            raise HTTPException(
                status_code=429, detail="Token limit exceeded after generation. Please upgrade your plan."
            )

        from app.core.cache_keys import CacheKeys
        from app.core.redis import redis_cache

        await redis_cache.delete_by_prefix(CacheKeys.user_sessions_prefix(user.id))
        await redis_cache.delete_by_prefix(CacheKeys.session_details_prefix(session_id))

        duration_ms = (time.perf_counter() - gen_start_time) * 1000.0
        provider_name = getattr(self.provider_registry.get_for_model(used_model, active_config), "name", "unknown")

        logger.info(
            "[AgentService] Chat completed: session_id=%s, provider='%s', model='%s', tokens=%d, duration=%.1fms, response='%s...'",
            session_id,
            provider_name,
            used_model,
            result.total_tokens,
            duration_ms,
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
        state: dict | None = None,
        request: Request | None = None,
    ) -> AsyncGenerator[str, None]:
        """Yields SSE-formatted chunks as the provider responds, then persists
        the full message and token usage once the stream is complete."""
        if not message_text or not message_text.strip():
            yield self._error_event("Message is empty.")
            return

        if state is not None:
            state["user_id"] = str(user.id)
            state["session_id"] = str(session_id) if session_id else None
            state["requested_model"] = requested_model
            state["requested_reasoning"] = requested_reasoning
            state["message_text"] = message_text
            state["attachment_ids"] = [str(a) for a in (attachment_ids or [])]
            state["completed"] = False
            state["handled"] = False

        try:
            is_new_session = session_id is None

            from app.ai.router.schemas import RequestContext
            from app.core.cache_keys import CacheKeys
            from app.core.redis import redis_cache
            from app.models.chats import MessageRole

            async def _shielded_prep():
                _active_config, _session_res = await asyncio.gather(
                    self.get_active_config(),
                    self._resolve_session(user, session_id, message_text, model=requested_model),
                )
                if len(_session_res) == 4:
                    _session, _session_id, _title, _needs_ai_worker = _session_res
                else:
                    _session, _session_id, _title = _session_res[:3]
                    _needs_ai_worker = False

                if _needs_ai_worker and getattr(_active_config, "enable_ai_title_generation", True):
                    _enqueue_task = asyncio.create_task(
                        self.enqueue_title_generation(_session_id, user.id, message_text, model=requested_model)
                    )
                    _fire_and_forget_tasks.add(_enqueue_task)
                    _enqueue_task.add_done_callback(_fire_and_forget_tasks.discard)

                if state is not None:
                    state["session_id"] = str(_session_id)

                if is_new_session:
                    await redis_cache.delete_by_prefix(CacheKeys.user_sessions_prefix(user.id))

                await self._quota_precheck(user, _active_config)

                _attachments = await self._prepare_attachments(user, _active_config, attachment_ids, _session_id)

                routing_context = RequestContext(
                    conversation_message_count=len(_session.messages) if _session and _session.messages else 0,
                    approximate_context_tokens=len(message_text) // 4,
                    has_attachments=bool(_attachments),
                    attachment_types=[a.type for a in _attachments],
                    user_id=str(user.id),
                    session_id=str(_session_id),
                )

                (
                    _model,
                    _reasoning,
                    _thinking_budget,
                    _tool_configs,
                    _fallbacks,
                ) = await self.validate_and_resolve_config(
                    requested_model=requested_model,
                    requested_reasoning=requested_reasoning,
                    message_text=message_text,
                    routing_mode=routing_mode,
                    context=routing_context,
                )

                if state is not None:
                    state["resolved_model"] = _model
                    state["resolved_reasoning"] = _reasoning

                _user_parts = [{"type": "text", "text": message_text}] if message_text else []
                for a in _attachments:
                    _user_parts.append(
                        {
                            "type": "file",
                            "attachment_id": str(a.id),
                            "filename": a.filename,
                            "contentType": a.mime_type,
                            "mediaType": a.mime_type,
                            "size": a.size,
                            "bucket": a.bucket,
                            "object_name": a.object_name,
                            "storage_uri": a.storage_uri,
                        }
                    )

                _user_msg = await self.chat_repo.add_message(
                    _session_id,
                    role=MessageRole.USER,
                    content=message_text,
                    attachment_ids=[str(a.id) for a in _attachments],
                    parts=_user_parts,
                )
                if state is not None:
                    state["user_message_id"] = str(_user_msg.id)

                async def _invalidate_caches():
                    await redis_cache.delete_by_prefix(CacheKeys.user_sessions_prefix(user.id))
                    await redis_cache.delete_by_prefix(CacheKeys.session_details_prefix(_session_id))

                _invalidate_task = asyncio.create_task(_invalidate_caches())
                _fire_and_forget_tasks.add(_invalidate_task)
                _invalidate_task.add_done_callback(_fire_and_forget_tasks.discard)

                return (
                    _active_config,
                    _session,
                    _session_id,
                    _title,
                    _attachments,
                    _model,
                    _reasoning,
                    _thinking_budget,
                    _tool_configs,
                    _fallbacks,
                    _user_msg,
                )

            (
                active_config,
                session,
                resolved_session_id,
                title,
                attachments,
                model,
                reasoning,
                thinking_budget,
                tool_configs,
                fallbacks,
                user_msg,
            ) = await asyncio.shield(_shielded_prep())
            session_id = resolved_session_id

            if state is not None:
                state["session_id"] = str(session_id)
                state["user_message_id"] = str(user_msg.id)
                state["resolved_model"] = model
                state["resolved_reasoning"] = reasoning
                state["attachment_ids"] = [str(a.id) for a in attachments]
        except (asyncio.CancelledError, GeneratorExit):
            # Client disconnected during prep
            if state is not None and not state.get("completed") and not state.get("handled"):
                _abandon_task = asyncio.create_task(handle_stream_abandonment(state, self))
                _fire_and_forget_tasks.add(_abandon_task)
                _abandon_task.add_done_callback(_fire_and_forget_tasks.discard)
            raise
        except Exception as prep_exc:
            if state is not None:
                state["handled"] = True
                state["failed"] = True
            if isinstance(prep_exc, HTTPException):
                detail = prep_exc.detail if isinstance(prep_exc.detail, str) else json.dumps(prep_exc.detail)
                code = "QUOTA_EXCEEDED" if prep_exc.status_code == 429 else "PREP_ERROR"
            else:
                detail = str(prep_exc)
                code = "PREP_ERROR"
            yield self._error_event(detail, code=code)
            return

        try:
            yield f"event: session\ndata: {json.dumps({'session_id': str(session_id), 'title': title})}\n\n"
            if title:
                yield f"event: chunk\ndata: {json.dumps({'type': 'title', 'title': title, 'session_id': str(session_id)})}\n\n"

            current_parts = await self._prepare_current_parts(message_text, attachments)
            contents = await self._build_contents(user, session, active_config, current_parts)
            generation_config = GenerationConfig(
                system_instruction=get_base_system_instructions(),
                thinking_budget=thinking_budget,
                include_thoughts=True,
                tool_configs=tool_configs,
            )

            models_to_attempt = [m for m in [model] + [m for m in fallbacks if m != model] if m and m.lower() != "auto"]
            used_model = model
            last_exc = None
            has_yielded_chunks = False
            full_response = ""
            full_thoughts = ""
            total_tokens = 0
            stream_start_time = time.perf_counter()
            ttft_ms: float | None = None

            collected_tools: dict[str, dict[str, Any]] = {}
            for attempt_idx, candidate_model in enumerate(models_to_attempt):
                full_response = ""
                full_thoughts = ""
                total_tokens = 0
                collected_tools = {}
                used_model = candidate_model
                last_exc = None

                logger.info(
                    "[AgentService] Invoking provider generate_stream for model='%s' with %d contents",
                    candidate_model,
                    len(contents),
                )

                try:
                    async for event in self.executor.generate_stream(candidate_model, contents, generation_config):
                        if (event.type in ("text", "reasoning")) and ttft_ms is None:
                            ttft_ms = (time.perf_counter() - stream_start_time) * 1000.0

                        if event.type == "text":
                            chunk = event.token or ""
                            full_response += chunk
                            if state is not None:
                                state["buffered_text"] = full_response
                            yield f"event: chunk\ndata: {json.dumps(self._serialize_event(event))}\n\n"
                            has_yielded_chunks = True
                        elif event.type == "reasoning":
                            full_thoughts += event.token or ""
                            if state is not None:
                                state["buffered_thoughts"] = full_thoughts
                            yield f"event: chunk\ndata: {json.dumps(self._serialize_event(event))}\n\n"
                            has_yielded_chunks = True
                        elif event.type == "tool_call":
                            call = event.tool_call
                            if call:
                                collected_tools[call.id] = {
                                    "type": "dynamic-tool",
                                    "toolName": call.name,
                                    "tool_name": call.name,
                                    "toolCallId": call.id,
                                    "tool_call_id": call.id,
                                    "state": "output-available",
                                    "input": call.args or {},
                                    "output": {},
                                }
                            yield f"event: chunk\ndata: {json.dumps(self._serialize_event(event))}\n\n"
                            has_yielded_chunks = True
                        elif event.type == "tool_result":
                            res = event.tool_result
                            if res:
                                if res.id in collected_tools:
                                    collected_tools[res.id]["output"] = res.output or {}
                                else:
                                    collected_tools[res.id] = {
                                        "type": "dynamic-tool",
                                        "toolName": res.name,
                                        "tool_name": res.name,
                                        "toolCallId": res.id,
                                        "tool_call_id": res.id,
                                        "state": "output-available",
                                        "input": {},
                                        "output": res.output or {},
                                    }
                            yield f"event: chunk\ndata: {json.dumps(self._serialize_event(event))}\n\n"
                            has_yielded_chunks = True
                        elif event.type == "usage" and event.total_tokens:
                            total_tokens = event.total_tokens
                            yield f"event: chunk\ndata: {json.dumps(self._serialize_event(event))}\n\n"

                    last_exc = None
                    break
                except (asyncio.CancelledError, GeneratorExit):
                    raise
                except Exception as exc:
                    last_exc = exc
                    await blacklist_model(candidate_model)
                    is_retryable = is_retryable_provider_error(exc)
                    logger.warning(
                        "[AgentService] Provider generate_stream failed for model='%s' (retryable=%s): %s",
                        candidate_model,
                        is_retryable,
                        exc,
                    )
                    if has_yielded_chunks or attempt_idx >= len(models_to_attempt) - 1 or not is_retryable:
                        break

            if not full_response.strip() and last_exc is None:
                logger.warning(
                    "[AgentService] Model '%s' generated thoughts but no text response. Triggering fallback recovery...",
                    used_model,
                )
                try:
                    recovery_config = GenerationConfig(
                        system_instruction=generation_config.system_instruction,
                        thinking_budget=0,
                        include_thoughts=False,
                        tool_configs=generation_config.tool_configs,
                    )
                    recovery_res = await self.executor.generate(used_model, contents, recovery_config)
                    if recovery_res and recovery_res.text:
                        full_response = recovery_res.text
                        yield f"event: chunk\ndata: {json.dumps(self._serialize_event(GenerationEvent(type='text', token=full_response)))}\n\n"
                        if recovery_res.total_tokens:
                            total_tokens = max(total_tokens, recovery_res.total_tokens)
                except Exception as rec_exc:
                    logger.error("[AgentService] Recovery generation failed: %s", rec_exc)

            final_parts: list[dict[str, Any]] = []
            if full_response.strip():
                final_parts.append({"type": "text", "text": full_response})

            # Extract and persist ONLY sources (do not persist thoughts or tool calls across reloads)
            extracted_sources: list[dict[str, str]] = []
            seen_source_urls: set[str] = set()
            for tool_item in collected_tools.values():
                tool_output = tool_item.get("output") or {}
                raw_sources = tool_output.get("sources") or []
                if isinstance(raw_sources, list):
                    for src in raw_sources:
                        if isinstance(src, dict):
                            url = src.get("uri") or src.get("url") or ""
                            if url and url not in seen_source_urls:
                                seen_source_urls.add(url)
                                title = src.get("title") or "Web Source"
                                extracted_sources.append({"title": title, "url": url, "uri": url})
            if extracted_sources:
                final_parts.append({"type": "sources", "sources": extracted_sources})

            if last_exc is not None:
                status, error_code, message = classify_provider_error(last_exc)

                logger.error(
                    "Stream generation failed for session %s with model %s: [%s] status=%d: %s",
                    session_id,
                    used_model,
                    error_code,
                    status,
                    last_exc,
                )
                if state is not None:
                    state["completed"] = True

                if full_response:
                    await self.chat_repo.add_message(
                        session_id,
                        role=MessageRole.AGENT,
                        content=full_response,
                        error=f"[{error_code}] {message}",
                        parts=final_parts,
                    )
                else:
                    await self.chat_repo.update_message(
                        session_id,
                        user_msg.id,
                        error=f"[{error_code}] {message}",
                    )

                yield self._error_event(message, error_code)
                return

            from app.core.cache_keys import CacheKeys
            from app.core.redis import redis_cache

            # Concurrently ensure agent response is stored, quota is charged, and cache is evicted
            try:
                await self.chat_repo.add_message(
                    session_id,
                    role=MessageRole.AGENT,
                    content=full_response,
                    parts=final_parts,
                )
            except Exception as e:
                logger.error("Failed to add agent message: %s", e)

            if state is not None:
                state["completed"] = True

            persist_results = await asyncio.gather(
                self._charge_usage(user, total_tokens, active_config),
                redis_cache.delete_by_prefix(CacheKeys.user_sessions_prefix(user.id)),
                redis_cache.delete_by_prefix(CacheKeys.session_details_prefix(session_id)),
                return_exceptions=True,
            )

            within_limit = persist_results[0] if isinstance(persist_results[0], bool) else True
            if not within_limit:
                yield self._error_event(
                    "Token limit reached after generation. Further messages are paused until quota resets.",
                    code="QUOTA_EXCEEDED",
                )
                return

            total_duration_ms = (time.perf_counter() - stream_start_time) * 1000.0
            provider_name = (
                getattr(self.provider_registry.get_for_model(used_model, active_config), "name", "unknown")
                if hasattr(self, "provider_registry")
                else "unknown"
            )

            logger.info(
                "[AgentService] Stream chat completed: session_id=%s, provider='%s', model='%s', tokens=%d, TTFT=%.1fms, total_duration=%.1fms, full_response='%s...'",
                session_id,
                provider_name,
                used_model,
                total_tokens,
                ttft_ms if ttft_ms is not None else 0.0,
                total_duration_ms,
                full_response[:80].replace("\n", " ") if full_response else "",
            )

            yield (
                "event: done\n"
                f"data: {json.dumps({'session_id': str(session_id), 'title': title, 'tokens_used': total_tokens, 'model': used_model, 'reasoning': reasoning})}\n\n"
            )
        except (asyncio.CancelledError, GeneratorExit):
            # Client disconnected anytime during the streaming lifecycle!
            logger.info(
                "Client disconnected during stream_chat for session %s. Relinquishing to background worker.",
                session_id,
            )
            if state is not None and not state.get("completed") and not state.get("handled"):
                _abandon_task = asyncio.create_task(handle_stream_abandonment(state, self))
                _fire_and_forget_tasks.add(_abandon_task)
                _abandon_task.add_done_callback(_fire_and_forget_tasks.discard)
            raise
        except HTTPException as exc:
            if state is not None:
                state["handled"] = True
                state["failed"] = True
            code = "QUOTA_EXCEEDED" if exc.status_code == 429 else "STREAM_ERROR"
            yield self._error_event(str(exc.detail), code=code)
            return
        except Exception as exc:
            if state is not None:
                state["handled"] = True
                state["failed"] = True
            logger.exception("[AgentService] Stream failed for session_id=%s", session_id)
            yield self._error_event(f"Internal error during stream: {exc}", code="STREAM_ERROR")
            return


async def handle_stream_abandonment(state: dict, agent_service: AgentService) -> None:
    """Handle premature stream disconnect by creating a GenerationDB and enqueuing a background task."""
    if not state or state.get("completed") or state.get("handled") or state.get("failed"):
        return
    state["handled"] = True

    user_id_str = state.get("user_id")
    if not user_id_str:
        return

    import uuid

    from app.core.cache_keys import CacheKeys
    from app.core.redis import redis_cache
    from app.models.chats import GenerationDB, GenerationStatus
    from app.schemas.task import GenerationTaskPayload

    user_id = uuid.UUID(user_id_str)
    session_id_str = state.get("session_id")
    if not session_id_str:
        user = await agent_service.user_repo.get_by_id(user_id_str)
        if not user:
            return
        _, session_id, _, _ = await agent_service._resolve_session(user, None, state.get("message_text", ""))
        session_id_str = str(session_id)
        state["session_id"] = session_id_str
    else:
        session_id = uuid.UUID(session_id_str)

    gen_id = uuid.uuid4()

    logger.info(
        "[AgentService] Stream abandoned by client for session %s. Lazily creating Generation %s & dispatching worker.",
        session_id,
        gen_id,
    )

    user_msg_id_str = state.get("user_message_id")
    user_msg_id = uuid.UUID(user_msg_id_str) if user_msg_id_str else None

    gen = GenerationDB(
        id=gen_id,
        user_id=user_id,
        session_id=session_id,
        user_message_id=user_msg_id,
        prompt=state.get("message_text"),
        status=GenerationStatus.QUEUED,
        requested_model=state.get("requested_model"),
        resolved_model=state.get("resolved_model") or state.get("model"),
        requested_reasoning=state.get("requested_reasoning"),
        resolved_reasoning=state.get("resolved_reasoning"),
        buffered_text=state.get("buffered_text", ""),
        buffered_thoughts=state.get("buffered_thoughts", ""),
    )

    try:
        await agent_service.generation_repo.create(gen)
        state["generation_id"] = str(gen.id)
    except Exception as e:
        logger.error("Failed to create GenerationDB record for abandoned stream: %s", e)
        return

    # Invalidate Redis caches so sidebar immediately shows active_generation_id
    try:
        await redis_cache.delete_by_prefix(CacheKeys.user_sessions_prefix(user_id))
        await redis_cache.delete_by_prefix(CacheKeys.session_details_prefix(session_id))
    except Exception as e:
        logger.warning("Failed to invalidate Redis cache on stream abandonment: %s", e)

    payload = GenerationTaskPayload(generation_id=str(gen.id))

    # Dispatch to Cloud Tasks (or fallback to local background execution if Cloud Tasks is unavailable/fails)
    enqueued = False
    if agent_service.tasks_dispatcher is not None:
        try:
            await agent_service.tasks_dispatcher.enqueue_generation_task(payload)
            enqueued = True
        except Exception as e:
            logger.warning("Cloud Tasks enqueue failed (%s). Falling back to background worker task.", e)

    if not enqueued:
        try:
            from app.repositories.chats import ChatRepository
            from app.repositories.config import ConfigRepository
            from app.repositories.generations import GenerationRepository
            from app.services.generation_worker import GenerationWorkerService

            gen_repo = agent_service.generation_repo or GenerationRepository(agent_service.db)
            worker = GenerationWorkerService(
                generation_repo=gen_repo,
                chat_repo=ChatRepository(agent_service.db),
                user_repo=UserRepository(agent_service.db),
                config_repo=ConfigRepository(agent_service.db),
                agent_service=agent_service,
            )
            _worker_task = asyncio.create_task(worker.execute_task(payload))
            _fire_and_forget_tasks.add(_worker_task)
            _worker_task.add_done_callback(_fire_and_forget_tasks.discard)
        except Exception as e:
            logger.error("Failed to execute local fallback worker for abandoned generation %s: %s", gen.id, e)
