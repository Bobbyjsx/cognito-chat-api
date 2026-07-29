import json
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime, timezone

from fastapi import HTTPException
from google.antigravity import Agent, CapabilitiesConfig, LocalAgentConfig, types

from app.core.config import settings
from app.models.chats import ChatResponse
from app.models.users import UserDB
from app.repositories.chats import ChatRepository
from app.repositories.users import UserRepository
from app.utils.datetime import ensure_utc
from app.utils.prompts import get_base_system_instructions


class AgentService:
    def __init__(self, chat_repo: ChatRepository, user_repo: UserRepository):
        self.chat_repo = chat_repo
        self.user_repo = user_repo

        self.agent_config = LocalAgentConfig(
            api_key=settings.gemini_api_key,
            system_instructions=get_base_system_instructions(),
            capabilities=CapabilitiesConfig(),
        )

    async def process_chat(self, user: UserDB, message_text: str, session_id: uuid.UUID | None = None) -> ChatResponse:

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

        # Build context from previous messages if session exists
        history_context = ""
        if session and session.messages:
            history_context = "\n".join([f"{m.role}: {m.content}" for m in session.messages])

        prompt = f"Chat History:\n{history_context}\n\nAgent, please respond to the latest user message: {message_text}"

        response_text = ""
        async with Agent(self.agent_config) as agent:
            response = await agent.chat(prompt)
            async for token in response:
                response_text += token

            total_tokens = 0
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage = response.usage_metadata
                total_tokens = getattr(usage, "total_token_count", 0)

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
    ) -> AsyncGenerator[str, None]:
        """Yields SSE-formatted chunks as the agent responds, then persists the
        full message and token usage once the stream is complete."""

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

        history_context = ""
        if session and session.messages:
            history_context = "\n".join([f"{m.role}: {m.content}" for m in session.messages])

        prompt = f"Chat History:\n{history_context}\n\nAgent, please respond to the latest user message: {message_text}"

        # Emit session_id first so the client can track it
        yield f"event: session\ndata: {json.dumps({'session_id': str(session_id)})}\n\n"

        full_response = ""
        total_tokens = 0
        async with Agent(self.agent_config) as agent:
            response = await agent.chat(prompt)
            async for chunk in response.chunks:
                if isinstance(chunk, types.Text):
                    full_response += chunk.text
                    yield f"data: {json.dumps({'type': 'text', 'token': chunk.text})}\n\n"
                elif isinstance(chunk, types.Thought):
                    yield f"data: {json.dumps({'type': 'thought', 'token': chunk.text})}\n\n"
                elif isinstance(chunk, types.ToolCall):
                    yield f"data: {json.dumps({'type': 'tool_call', 'name': chunk.name})}\n\n"

            if hasattr(response, "usage_metadata") and response.usage_metadata:
                total_tokens = getattr(response.usage_metadata, "total_token_count", 0)

        # Persist after stream completes
        await self.chat_repo.add_message(session_id, role="agent", content=full_response)
        if total_tokens > 0:
            within_limit = await self.user_repo.atomic_increment_if_within_limit(user.id, total_tokens)
            if not within_limit:
                yield f"event: error\ndata: {json.dumps({'detail': 'Token limit exceeded after generation. Please upgrade your plan.'})}\n\n"
                return

        yield f"event: done\ndata: {json.dumps({'session_id': str(session_id), 'tokens_used': total_tokens})}\n\n"
