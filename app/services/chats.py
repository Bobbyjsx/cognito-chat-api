from collections import deque
from datetime import datetime, timezone
import json
import logging
import uuid
from typing import AsyncGenerator

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

logger = logging.getLogger(__name__)

SAFE_GENERATION_ERROR = "Model generation failed. Please try again."


def get_base_system_instructions() -> str:
    """Standard system instructions for Gemini chat sessions."""
    return (
        "You are Cognito, an advanced AI assistant created to be helpful, concise, and clear. "
        "Format responses cleanly with Markdown when applicable."
    )


class AgentService:
    def __init__(
        self,
        chat_repo: ChatRepository,
        user_repo: UserRepository,
        config_repo: ConfigRepository,
        client: genai.Client | None = None,
    ):
        self.chat_repo = chat_repo
        self.user_repo = user_repo
        self.config_repo = config_repo

        if client is not None:
            self.client = client
        elif settings.gemini_api_key:
            self.client = genai.Client(api_key=settings.gemini_api_key)
        else:
            self.client = genai.Client()

    async def get_active_config(self) -> AppConfigDB:
        cfg = await self.config_repo.get_config()
        if not cfg:
            cfg = AppConfigDB()
            await self.config_repo.save_config(cfg)
        return cfg

    async def validate_and_resolve_config(
        self,
        requested_model: str | None = None,
        requested_reasoning: str | None = None,
    ) -> tuple[str, types.ThinkingConfig | None, str, list[types.Tool] | None]:

        cfg = await self.get_active_config()

        if not cfg.enable_text_generation:
            raise HTTPException(status_code=403, detail="Text generation is currently disabled by admin.")

        model = requested_model or cfg.default_text_model
        if model not in cfg.allowed_text_models:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model}' is not in the allowed text models list: {cfg.allowed_text_models}",
            )

        allowed_for_model = cfg.model_reasoning_modes.get(model, cfg.allowed_reasoning_levels)

        reasoning = requested_reasoning or cfg.default_reasoning_level
        if reasoning not in cfg.allowed_reasoning_levels or reasoning not in allowed_for_model:
            reasoning = "none"

        thinking_config: types.ThinkingConfig | None = None
        if reasoning != "none":
            budget_map = {
                "low": 1024,
                "medium": 4096,
                "high": 8192,
            }
            budget = budget_map.get(reasoning, 2048)
            thinking_config = types.ThinkingConfig(thinking_budget=budget)

        tools: list[types.Tool] | None = None
        if cfg.allowed_tools and len(cfg.allowed_tools) > 0:
            tool_list = []
            if "code_execution" in cfg.allowed_tools:
                tool_list.append(types.Tool(code_execution=types.CodeExecution()))

            if tool_list:
                tools = tool_list

        return model, thinking_config, reasoning, tools

    async def _generate_title(self, first_message: str) -> str:
        """Generates a concise title (3-5 words) for a new chat session."""
        try:
            prompt = (
                "Summarize the following user prompt into a short, descriptive chat title (maximum 5 words, no quotes, plain text):\n\n"
                f"{first_message[:300]}"
            )
            resp = await self.client.aio.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=prompt,
            )
            title = (resp.text or "").strip().strip('"').strip("'")
            if title and len(title) <= 60:
                return title
        except Exception:
            logger.warning("Fast title generation failed, falling back to message snippet.")

        words = first_message.strip().split()
        return " ".join(words[:5]) if words else "New Chat"

    def _build_generate_config(
        self,
        thinking_config: types.ThinkingConfig | None,
        tools: list[types.Tool] | None,
    ) -> types.GenerateContentConfig:
        system_instructions = get_base_system_instructions()
        return types.GenerateContentConfig(
            system_instruction=system_instructions,
            thinking_config=thinking_config,
            tools=tools,
        )

    def _build_contents(self, history_messages: list, current_message: str) -> list[types.Content]:
        contents: list[types.Content] = []
        for msg in history_messages:
            role = "user" if msg.role == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg.content)]))
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=current_message)]))
        return contents

    def _extract_stream_events(self, chunk, code_tool_ids: deque[str] | None = None) -> list[dict]:
        events: list[dict] = []
        if code_tool_ids is None:
            code_tool_ids = deque()

        candidates = getattr(chunk, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if not content:
                continue

            parts = getattr(content, "parts", None) or []
            for part in parts:
                text = getattr(part, "text", None)
                thought = getattr(part, "thought", None)
                executable_code = getattr(part, "executable_code", None)
                code_execution_result = getattr(part, "code_execution_result", None)

                if thought:
                    thought_str = text if text else (thought if isinstance(thought, str) else "")
                    if thought_str:
                        events.append({"type": "reasoning", "token": thought_str})
                elif text:
                    events.append({"type": "text", "token": text})

                if executable_code:
                    tool_call_id = f"call_{uuid.uuid4().hex[:8]}"
                    code_tool_ids.append(tool_call_id)
                    language = getattr(executable_code, "language", None)
                    lang_str = str(language).lower() if language else "python"
                    events.append(
                        {
                            "type": "tool_call",
                            "tool_name": "code_execution",
                            "tool_call_id": tool_call_id,
                            "input": {
                                "language": lang_str,
                                "code": getattr(executable_code, "code", ""),
                            },
                        }
                    )

                if code_execution_result:
                    tool_call_id = code_tool_ids.popleft() if code_tool_ids else f"call_{uuid.uuid4().hex[:8]}"
                    outcome = getattr(code_execution_result, "outcome", None)
                    events.append(
                        {
                            "type": "tool_result",
                            "tool_name": "code_execution",
                            "tool_call_id": tool_call_id,
                            "output": {
                                "outcome": str(outcome) if outcome else "OUTCOME_OK",
                                "output": getattr(code_execution_result, "output", ""),
                            },
                        }
                    )

        return events

    async def process_chat(
        self,
        user: UserDB,
        message_text: str,
        session_id: uuid.UUID | None = None,
        requested_model: str | None = None,
        requested_reasoning: str | None = None,
    ) -> ChatResponse:
        """Non-streaming processing of chat messages."""
        if not message_text or not message_text.strip():
            raise HTTPException(status_code=400, detail="Message is empty.")

        model, thinking_config, reasoning, tools = await self.validate_and_resolve_config(
            requested_model, requested_reasoning
        )

        now = datetime.now(timezone.utc)
        reset_at = ensure_utc(user.reset_at)
        weekly_reset_at = ensure_utc(user.weekly_reset_at)

        is_6h_expired = reset_at is None or reset_at <= now
        is_weekly_expired = weekly_reset_at is None or weekly_reset_at <= now

        effective_6h = 0 if is_6h_expired else user.tokens_used_6h
        effective_weekly = 0 if is_weekly_expired else user.tokens_used_weekly

        if effective_6h >= user.token_limit_6h:
            reset_str = reset_at.isoformat() if reset_at else ""
            raise HTTPException(status_code=429, detail=f"6-hour token limit reached. Resets at {reset_str}")
        if effective_weekly >= user.token_limit_weekly:
            reset_str = weekly_reset_at.isoformat() if weekly_reset_at else ""
            raise HTTPException(status_code=429, detail=f"Weekly token limit reached. Resets at {reset_str}")

        title = None
        if session_id is None:
            title = await self._generate_title(message_text)
            session = await self.chat_repo.create_session(user_id=user.id, title=title)
            session_id = session.id
        else:
            session = await self.chat_repo.get_session(session_id, user_id=user.id)
            if not session:
                raise HTTPException(status_code=44, detail=f"Session {session_id} not found.")
            if not session.title:
                title = await self._generate_title(message_text)
                await self.chat_repo.update_session_title(session_id, title)
            else:
                title = session.title

        await self.chat_repo.add_message(session_id, role="user", content=message_text)

        history_messages = session.messages if session and session.messages else []
        contents = self._build_contents(history_messages, message_text)
        config = self._build_generate_config(thinking_config, tools)

        try:
            response = await self.client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            response_text = response.text or ""
            total_tokens = getattr(response.usage_metadata, "total_token_count", 0) if response.usage_metadata else 0
        except Exception:
            logger.exception("Model generation failed for session %s", session_id)
            await self.chat_repo.add_message(
                session_id, role="agent", content="", error=SAFE_GENERATION_ERROR
            )
            raise HTTPException(status_code=500, detail=SAFE_GENERATION_ERROR) from None

        await self.chat_repo.add_message(session_id, role="agent", content=response_text)
        if total_tokens > 0:
            within_limit = await self.user_repo.atomic_increment_if_within_limit(user.id, total_tokens)
            if not within_limit:
                raise HTTPException(
                    status_code=429, detail="Token limit exceeded after generation. Please upgrade your plan."
                )

        return ChatResponse(session_id=session_id, title=title, response=response_text)

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

        if not message_text or not message_text.strip():
            yield f"event: error\ndata: {json.dumps({'detail': 'Message is empty.'})}\n\n"
            return

        try:
            model, thinking_config, reasoning, tools = await self.validate_and_resolve_config(
                requested_model, requested_reasoning
            )
        except HTTPException as e:
            yield f"event: error\ndata: {json.dumps({'detail': e.detail})}\n\n"
            return

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

        session_title = None
        if session_id is None:
            session_title = await self._generate_title(message_text)
            session = await self.chat_repo.create_session(user_id=user.id, title=session_title)
            session_id = session.id
        else:
            session = await self.chat_repo.get_session(session_id, user_id=user.id)
            if not session:
                yield f"event: error\ndata: {json.dumps({'detail': f'Session {session_id} not found.'})}\n\n"
                return
            if not session.title:
                session_title = await self._generate_title(message_text)
                await self.chat_repo.update_session_title(session_id, session_title)
            else:
                session_title = session.title

        await self.chat_repo.add_message(session_id, role="user", content=message_text)

        history_messages = session.messages if session and session.messages else []
        contents = self._build_contents(history_messages, message_text)
        config = self._build_generate_config(thinking_config, tools)

        yield f"event: session\ndata: {json.dumps({'session_id': str(session_id), 'title': session_title})}\n\n"

        full_response = ""
        total_tokens = 0
        code_tool_ids: deque[str] = deque()

        try:
            response_stream = await self.client.aio.models.generate_content_stream(
                model=model,
                contents=contents,
                config=config,
            )

            async for chunk in response_stream:
                events = self._extract_stream_events(chunk, code_tool_ids)
                for event in events:
                    if event.get("type") == "text" and "token" in event:
                        full_response += event["token"]
                    yield f"event: chunk\ndata: {json.dumps(event)}\n\n"

                if getattr(chunk, "usage_metadata", None):
                    total_tokens = getattr(chunk.usage_metadata, "total_token_count", total_tokens)
        except Exception:
            logger.exception("Model generation failed for session %s", session_id)
            await self.chat_repo.add_message(
                session_id, role="agent", content=full_response, error=SAFE_GENERATION_ERROR
            )
            yield f"event: error\ndata: {json.dumps({'detail': SAFE_GENERATION_ERROR})}\n\n"
            return

        await self.chat_repo.add_message(session_id, role="agent", content=full_response)
        if total_tokens > 0:
            within_limit = await self.user_repo.atomic_increment_if_within_limit(user.id, total_tokens)
            if not within_limit:
                yield f"event: error\ndata: {json.dumps({'detail': 'Token limit exceeded after generation. Please upgrade your plan.'})}\n\n"
                return

        yield (
            "event: done\n"
            f"data: {json.dumps({'session_id': str(session_id), 'title': session_title, 'tokens_used': total_tokens, 'model': model, 'reasoning': reasoning})}\n\n"
        )
