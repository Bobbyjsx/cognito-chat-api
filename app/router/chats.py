import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from google.cloud.firestore_v1.async_client import AsyncClient

from app.api.dependencies import get_current_user
from app.database import get_db
from app.models.chats import ChatRequest, ChatResponse, ChatSessionListSchema, ChatSessionSchema
from app.models.users import UserDB
from app.repositories.chats import ChatRepository
from app.repositories.config import ConfigRepository
from app.repositories.users import UserRepository
from app.services.chats import AgentService

router = APIRouter(prefix="/agent", tags=["agent"])


def get_agent_service(db: AsyncClient = Depends(get_db)) -> AgentService:
    chat_repo = ChatRepository(db)
    user_repo = UserRepository(db)
    config_repo = ConfigRepository(db)
    return AgentService(chat_repo=chat_repo, user_repo=user_repo, config_repo=config_repo)


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
        )
        return response
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException as e:
        raise e
    except Exception as e:
        import traceback

        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Error: {e!s}\nTraceback: {tb}")


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
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disables nginx / Cloud Run proxy buffering
        },
    )


@router.get("/sessions", response_model=list[ChatSessionListSchema])
async def list_sessions(current_user: UserDB = Depends(get_current_user), db: AsyncClient = Depends(get_db)):
    repo = ChatRepository(db)
    sessions = await repo.get_user_sessions(current_user.id)
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


@router.post("/sessions/{session_id}/read", status_code=200)
async def mark_as_read(
    session_id: uuid.UUID,
    current_user: UserDB = Depends(get_current_user),
    db: AsyncClient = Depends(get_db),
):
    repo = ChatRepository(db)
    # Check if session exists and belongs to user
    session = await repo.get_session(session_id, current_user.id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    await repo.mark_session_read(session_id)
    return {"message": "Session marked as read"}
