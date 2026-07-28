import uuid

from fastapi import HTTPException
from google.antigravity import Agent, CapabilitiesConfig, LocalAgentConfig

from app.core.config import settings
from app.models.chats import ChatResponse
from app.models.users import UserDB
from app.repositories.chats import ChatRepository
from app.repositories.users import UserRepository
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

        # Check token limits
        if user.tokens_used >= user.token_limit:
            raise HTTPException(
                status_code=403,
                detail="Token limit exceeded. Please upgrade your plan.",
            )

        if session_id is None:
            # Create a new session in PostgreSQL
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

            # Extract token usage from the response
            # Note: the usage metadata format varies, we attempt to sum standard counts
            total_tokens = 0
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage = response.usage_metadata
                total_tokens = getattr(usage, "total_token_count", 0)

            if total_tokens > 0:
                await self.user_repo.update_token_usage(user.id, total_tokens)

        # Save agent response
        await self.chat_repo.add_message(session_id, role="agent", content=response_text)

        return ChatResponse(session_id=session_id, response=response_text)
