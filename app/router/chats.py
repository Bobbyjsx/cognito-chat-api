import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from google.cloud.firestore_v1.async_client import AsyncClient

from app.api.dependencies import (
    get_current_user,
    get_provider,
    get_storage_backend,
    get_tool_registry,
)
from app.database import get_db
from app.models.chats import ChatRequest, ChatResponse, ChatSessionListSchema, ChatSessionSchema
from app.models.users import UserDB
from app.providers.base import BaseProvider
from app.repositories.attachments import AttachmentRepository
from app.repositories.chats import ChatRepository
from app.repositories.config import ConfigRepository
from app.repositories.users import UserRepository
from app.services.attachments import AttachmentService
from app.services.chats import AgentService
from app.storage.base import StorageBackend
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])


def get_agent_service(
    db: AsyncClient = Depends(get_db),
    provider: BaseProvider = Depends(get_provider),
    storage: StorageBackend = Depends(get_storage_backend),
    registry: ToolRegistry = Depends(get_tool_registry),
) -> AgentService:
    chat_repo = ChatRepository(db)
    user_repo = UserRepository(db)
    config_repo = ConfigRepository(db)
    attachment_service = AttachmentService(AttachmentRepository(db), storage, provider)
    executor = ToolExecutor(registry, provider)
    return AgentService(
        chat_repo=chat_repo,
        user_repo=user_repo,
        config_repo=config_repo,
        attachment_service=attachment_service,
        provider=provider,
        registry=registry,
        executor=executor,
    )


@router.post("/chat", response_model=ChatResponse)
async def chat_with_agent(
    request: ChatRequest,
    session_id: uuid.UUID | None = None,
    current_user: UserDB = Depends(get_current_user),
    agent_service: AgentService = Depends(get_agent_service),
):
    try:
        response = await agent_service.process_chat(
            user=current_user,
            message_text=request.message,
            session_id=session_id,
            requested_model=request.model,
            requested_reasoning=request.reasoning,
            attachment_ids=request.attachments,
        )
        return response
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unexpected error in /agent/chat")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.") from None


@router.post("/chat/stream", summary="Stream a chat response via SSE")
async def stream_chat_with_agent(
    request: ChatRequest,
    session_id: uuid.UUID | None = None,
    current_user: UserDB = Depends(get_current_user),
    agent_service: AgentService = Depends(get_agent_service),
):
    return StreamingResponse(
        agent_service.stream_chat(
            user=current_user,
            message_text=request.message,
            session_id=session_id,
            requested_model=request.model,
            requested_reasoning=request.reasoning,
            attachment_ids=request.attachments,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/sessions", response_model=list[ChatSessionListSchema])
async def list_sessions(
    q: str | None = None,
    current_user: UserDB = Depends(get_current_user),
    db: AsyncClient = Depends(get_db),
):
    repo = ChatRepository(db)
    sessions = await repo.get_user_sessions(current_user.id, search_query=q)
    return sessions


@router.get("/sessions/{session_id}", response_model=ChatSessionSchema)
async def get_session(
    session_id: uuid.UUID,
    current_user: UserDB = Depends(get_current_user),
    db: AsyncClient = Depends(get_db),
):
    repo = ChatRepository(db)
    session = await repo.get_session(session_id, current_user.id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    await repo.mark_session_read(session_id)
    session.read_status = "read"

    return session


@router.delete("/sessions/{session_id}", status_code=200)
@router.post("/sessions/{session_id}/delete", status_code=200)
async def delete_session(
    session_id: uuid.UUID,
    current_user: UserDB = Depends(get_current_user),
    db: AsyncClient = Depends(get_db),
):
    repo = ChatRepository(db)
    success = await repo.soft_delete_session(session_id, current_user.id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"message": "Session deleted successfully"}


@router.post("/sessions/{session_id}/read", status_code=200)
async def mark_as_read(
    session_id: uuid.UUID,
    current_user: UserDB = Depends(get_current_user),
    db: AsyncClient = Depends(get_db),
):
    repo = ChatRepository(db)
    session = await repo.get_session(session_id, current_user.id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    await repo.mark_session_read(session_id)
    return {"message": "Session marked as read"}
