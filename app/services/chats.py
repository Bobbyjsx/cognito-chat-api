import json
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime, timezone

from fastapi import HTTPException
from google import genai
from google.genai import types

from app.core.config import settings
from app.models.chats import ChatResponse
from app.models.config import AppConfigDB
from app.models.users import UserDB
from app.repositories.chats import ChatRepository
from app.repositories.config import ConfigRepository
from app.repositories.users import UserRepository
from app.utils.datetime import ensure_utc
from app.utils.prompts import get_base_system_instructions


class AgentService:
    def __init__(
        self,
        chat_repo: ChatRepository,
        user_repo: UserRepository,
        config_repo: ConfigRepository,
    ):
        self.chat_repo = chat_repo
        self.user_repo = user_repo
        self.config_repo = config_repo
        self.client = genai.Client(api_key=settings.gemini_api_key)

    async def validate_and_resolve_config(
        self,
        requested_model: str | None,
        requested_reasoning: str | None,
    ) -> tuple[str, types.ThinkingConfig | None]:
        """Fetches system configuration from Firestore and validates requested model and reasoning.

        Returns (resolved_model, thinking_config).
        Raises HTTPException 400 if model or reasoning level is not allowed.
        """
        sys_config: AppConfigDB = await self.config_repo.get_config()

        # 1. Resolve and validate text model
        model = requested_model if requested_model else sys_config.default_text_model
        if model not in sys_config.allowed_text_models:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model}' is not allowed or supported. Allowed text models: {sys_config.allowed_text_models}",
            )

        # 2. Resolve and validate reasoning level
        reasoning = requested_reasoning.lower() if requested_reasoning else sys_config.default_reasoning_level.lower()
        allowed_reasoning_lower = [r.lower() for r in sys_config.allowed_reasoning_levels]
        if reasoning not in allowed_reasoning_lower:
            raise HTTPException(
                status_code=400,
                detail=f"Reasoning level '{reasoning}' is not allowed. Allowed reasoning levels: {sys_config.allowed_reasoning_levels}",
            )

        # 3. Construct Gemini ThinkingConfig
        thinking_config = None
        if reasoning == "none":
            thinking_config = types.ThinkingConfig(thinking_budget=0)
        elif reasoning in ("minimal", "low", "medium", "high"):
            thinking_config = types.ThinkingConfig(thinking_level=reasoning.upper())

        return model, thinking_config

    def _build_contents(self, history_messages: list, current_message: str) -> list[types.Content]:
        contents: list[types.Content] = []
        for m in history_messages:
            role = "user" if m.role == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=m.content)]))
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=current_message)]))
        return contents

    async def process_chat(
        self,
        user: UserDB,
        message_text: str,
        session_id: uuid.UUID | None = None,
        requested_model: str | None = None,
        requested_reasoning: str | None = None,
    ) -> ChatResponse:
        # Validate model and reasoning against Firestore config
        model, thinking_config = await self.validate_and_resolve_config(
            requested_model, requested_reasoning
        )

        # Check token limits before hitting the model
        now = datetime.now(timezone.utc)
        reset_at = ensure_utc(user.reset_at)
        weekly_reset_at = ensure_utc(user.weekly_reset_at)

        is_6h_expired = reset_at is None or reset_at <= now
        is_weekly_expired = weekly_reset_at is None or weekly_reset_at <= now

        effective_6h = 0 if is_6h_expired else user.tokens_used_6h
        effective_weekly = 0 if is_weekly_expired else user.tokens_used_weekly

        if effective_6h >= user.token_limit_6h:
            reset_str = reset_at.isoformat() if reset_at else ""
            raise HTTPException(status_code=403, detail=f"6-hour token limit reached. Resets at {reset_str}")
        if effective_weekly >= user.token_limit_weekly:
            reset_str = weekly_reset_at.isoformat() if weekly_reset_at else ""
            raise HTTPException(status_code=403, detail=f"Weekly token limit reached. Resets at {reset_str}")

        if session_id is None:
            session = await self.chat_repo.create_session(user_id=user.id)
            session_id = session.id
        else:
            session = await self.chat_repo.get_session(session_id, user_id=user.id)
            if not session:
                raise ValueError(f"Session {session_id} not found in database for this user.")

        # Save user message
        await self.chat_repo.add_message(session_id, role="user", content=message_text)

        history_messages = session.messages if session and session.messages else []
        contents = self._build_contents(history_messages, message_text)

        config = types.GenerateContentConfig(
            system_instruction=get_base_system_instructions(),
            thinking_config=thinking_config,
        )

        response = await self.client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

        response_text = response.text or ""
        total_tokens = 0
        if response.usage_metadata:
            total_tokens = getattr(response.usage_metadata, "total_token_count", 0)

        if total_tokens > 0:
            within_limit = await self.user_repo.atomic_increment_if_within_limit(user.id, total_tokens)
            if not within_limit:
                raise HTTPException(status_code=403, detail="Token limit exceeded. Please upgrade your plan.")

        # Save agent response
        await self.chat_repo.add_message(session_id, role="agent", content=response_text)

        return ChatResponse(session_id=session_id, response=response_text)

    async def stream_chat(
        self,
        user: UserDB,
        message_text: str,
        session_id: uuid.UUID | None = None,
        requested_model: str | None = None,
        requested_reasoning: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """Yields SSE-formatted chunks as Gemini responds, then persists the
        full message and token usage once the stream is complete."""

        # Validate model and reasoning against Firestore config
        try:
            model, thinking_config = await self.validate_and_resolve_config(
                requested_model, requested_reasoning
            )
        except HTTPException as e:
            yield f"event: error\ndata: {json.dumps({'detail': e.detail})}\n\n"
            return

        # Check token limits before hitting the model
        now = datetime.now(timezone.utc)
        reset_at = ensure_utc(user.reset_at)
        weekly_reset_at = ensure_utc(user.weekly_reset_at)

        is_6h_expired = reset_at is None or reset_at <= now
        is_weekly_expired = weekly_reset_at is None or weekly_reset_at <= now

        effective_6h = 0 if is_6h_expired else user.tokens_used_6h
        effective_weekly = 0 if is_weekly_expired else user.tokens_used_weekly

        if effective_6h >= user.token_limit_6h:
            reset_str = reset_at.isoformat() if reset_at else ""
            yield f"event: error\ndata: {json.dumps({'detail': f'6-hour token limit reached. Resets at {reset_str}'})}\n\n"
            return
        if effective_weekly >= user.token_limit_weekly:
            reset_str = weekly_reset_at.isoformat() if weekly_reset_at else ""
            yield f"event: error\ndata: {json.dumps({'detail': f'Weekly token limit reached. Resets at {reset_str}'})}\n\n"
            return

        if session_id is None:
            session = await self.chat_repo.create_session(user_id=user.id)
            session_id = session.id
        else:
            session = await self.chat_repo.get_session(session_id, user_id=user.id)
            if not session:
                yield f"event: error\ndata: {json.dumps({'detail': f'Session {session_id} not found.'})}\n\n"
                return

        await self.chat_repo.add_message(session_id, role="user", content=message_text)

        history_messages = session.messages if session and session.messages else []
        contents = self._build_contents(history_messages, message_text)

        config = types.GenerateContentConfig(
            system_instruction=get_base_system_instructions(),
            thinking_config=thinking_config,
        )

        # Emit session_id first so the client can track it
        yield f"event: session\ndata: {json.dumps({'session_id': str(session_id)})}\n\n"

        full_response = ""
        total_tokens = 0
        response_stream = await self.client.aio.models.generate_content_stream(
            model=model,
            contents=contents,
            config=config,
        )

        async for chunk in response_stream:
            if chunk.text:
                full_response += chunk.text
                yield f"data: {json.dumps({'type': 'text', 'token': chunk.text})}\n\n"
            if chunk.usage_metadata:
                total_tokens = getattr(chunk.usage_metadata, "total_token_count", total_tokens)

        # Persist after stream completes
        await self.chat_repo.add_message(session_id, role="agent", content=full_response)
        if total_tokens > 0:
            within_limit = await self.user_repo.atomic_increment_if_within_limit(user.id, total_tokens)
            if not within_limit:
                yield f"event: error\ndata: {json.dumps({'detail': 'Token limit exceeded after generation. Please upgrade your plan.'})}\n\n"
                return

        yield f"event: done\ndata: {json.dumps({'session_id': str(session_id), 'tokens_used': total_tokens})}\n\n"
