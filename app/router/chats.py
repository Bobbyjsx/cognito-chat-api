import asyncio
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from google.cloud.firestore_v1.async_client import AsyncClient

from app.api.dependencies import (
    get_current_user,
    get_provider,
    get_provider_registry,
    get_smart_router,
    get_storage_backend,
    get_tasks_dispatcher,
    get_tool_registry,
)
from app.database import get_db
from app.integrations.cloud_tasks import CloudTasksDispatcher
from app.models.chats import ChatRequest, ChatResponse, ChatSessionListSchema, ReadStatus
from app.models.pagination import PaginatedResponse
from app.models.users import UserDB
from app.providers.base import BaseProvider
from app.repositories.attachments import AttachmentRepository
from app.repositories.chats import ChatRepository
from app.repositories.config import ConfigRepository
from app.repositories.generations import GenerationRepository
from app.repositories.users import UserRepository
from app.services.attachments import AttachmentService
from app.services.chats import AgentService
from app.storage.base import StorageBackend
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])


class AgentStreamingResponse(StreamingResponse):
    def __init__(self, content, state: dict, agent_service, **kwargs):
        super().__init__(content, **kwargs)
        self.state = state
        self.agent_service = agent_service

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            if not self.state.get("completed") and not self.state.get("handled"):
                from app.services.chats import handle_stream_abandonment

                asyncio.create_task(handle_stream_abandonment(self.state, self.agent_service))


def get_agent_service(
    db: AsyncClient = Depends(get_db),
    provider: BaseProvider = Depends(get_provider),
    provider_registry=Depends(get_provider_registry),
    storage: StorageBackend = Depends(get_storage_backend),
    registry: ToolRegistry = Depends(get_tool_registry),
    smart_router=Depends(get_smart_router),
    dispatcher: CloudTasksDispatcher | None = Depends(get_tasks_dispatcher),
) -> AgentService:
    chat_repo = ChatRepository(db)
    user_repo = UserRepository(db)
    config_repo = ConfigRepository(db)
    generation_repo = GenerationRepository(db)
    attachment_service = AttachmentService(AttachmentRepository(db), storage, provider)
    executor = ToolExecutor(registry, provider_registry)
    return AgentService(
        chat_repo=chat_repo,
        user_repo=user_repo,
        config_repo=config_repo,
        attachment_service=attachment_service,
        provider=provider,
        provider_registry=provider_registry,
        registry=registry,
        executor=executor,
        router=smart_router,
        generation_repo=generation_repo,
        tasks_dispatcher=dispatcher,
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
            routing_mode=request.routing_mode,
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
    fastapi_req: Request,
    session_id: uuid.UUID | None = None,
    current_user: UserDB = Depends(get_current_user),
    agent_service: AgentService = Depends(get_agent_service),
):
    state = {"generation_id": None, "completed": False}
    return AgentStreamingResponse(
        agent_service.stream_chat(
            request=fastapi_req,
            user=current_user,
            message_text=request.message,
            session_id=session_id,
            requested_model=request.model,
            requested_reasoning=request.reasoning,
            attachment_ids=request.attachments,
            routing_mode=request.routing_mode,
            state=state,
        ),
        state=state,
        agent_service=agent_service,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/sessions", response_model=PaginatedResponse[ChatSessionListSchema])
async def list_sessions(
    q: str | None = None,
    limit: int = 10,
    offset: int = 0,
    current_user: UserDB = Depends(get_current_user),
    db: AsyncClient = Depends(get_db),
):
    from app.core.cache_keys import CacheKeys
    from app.core.redis import redis_cache
    from app.repositories.generations import GenerationRepository

    gen_repo = GenerationRepository(db)
    repo = ChatRepository(db)

    # Cache only the raw session list (invalidated on message/session events)
    cache_key = CacheKeys.user_sessions(current_user.id, limit, offset, q)
    cached_sessions = await redis_cache.get(cache_key)

    if cached_sessions:
        sessions_data = cached_sessions["sessions"]
        has_more = cached_sessions["has_more"]
        total = cached_sessions["total"]
        active_gens = await gen_repo.get_active_generations_for_user(current_user.id)
    else:
        (sessions_res, active_gens) = await asyncio.gather(
            repo.get_user_sessions(current_user.id, search_query=q, limit=limit, offset=offset),
            gen_repo.get_active_generations_for_user(current_user.id),
        )
        sessions, has_more, total = sessions_res
        sessions_data = [s.model_dump(mode="json") for s in sessions]
        await redis_cache.set(
            cache_key,
            {"sessions": sessions_data, "has_more": has_more, "total": total},
            expire=300,
        )

    # Merge active_generation_id into each session item
    items = []
    for s_dict in sessions_data:
        merged = dict(s_dict)
        sid = str(s_dict.get("id", ""))
        if sid in active_gens:
            merged["active_generation_id"] = active_gens[sid]
        else:
            merged["active_generation_id"] = None
        items.append(ChatSessionListSchema(**merged))

    return PaginatedResponse(items=items, total=total, limit=limit, offset=offset, has_more=has_more)


@router.get("/sessions/{session_id}")
async def get_session(
    session_id: uuid.UUID,
    response: Response,
    limit: int = 10,
    offset: int = 0,
    current_user: UserDB = Depends(get_current_user),
    db: AsyncClient = Depends(get_db),
):
    from app.core.cache_keys import CacheKeys
    from app.core.redis import redis_cache
    from app.repositories.generations import GenerationRepository

    gen_repo = GenerationRepository(db)
    repo = ChatRepository(db)

    cache_key = CacheKeys.session_details(session_id, limit, offset)

    # Concurrently fetch active generation status and session data from Firestore
    (active_gen, session_res) = await asyncio.gather(
        gen_repo.get_active_generation(session_id),
        repo.get_session(session_id, current_user.id, limit=limit, offset=offset),
    )
    session, has_more = session_res
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.read_status != ReadStatus.READ:
        session.read_status = ReadStatus.READ
        await repo.mark_session_read(session_id)
        await redis_cache.delete_by_prefix(CacheKeys.user_sessions_prefix(current_user.id))

    messages = session.messages or []
    session.messages = []

    result = {
        "session": session.model_dump(mode="json"),
        "messages": {
            "items": [msg.model_dump(mode="json") for msg in messages],
            "limit": limit,
            "offset": offset,
            "has_more": has_more,
        },
        "active_generation_id": str(active_gen.id) if active_gen else None,
    }

    if active_gen:
        # Active generation in progress: force no-cache for browser and intermediate proxies
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
    else:
        # Normal session: cache in Redis and allow standard caching
        await redis_cache.set(cache_key, result, expire=300)

    return result


@router.delete("/sessions/{session_id}", status_code=200)
@router.post("/sessions/{session_id}/delete", status_code=200)
async def delete_session(
    session_id: uuid.UUID,
    current_user: UserDB = Depends(get_current_user),
    db: AsyncClient = Depends(get_db),
):
    repo = ChatRepository(db)
    gen_repo = GenerationRepository(db)

    success = await repo.soft_delete_session(session_id, current_user.id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or already deleted")

    # Cancel any in-flight live or background generations for this session
    asyncio.create_task(
        gen_repo.cancel_active_generations_for_session(session_id, reason="Session was deleted by user")
    )

    from app.core.cache_keys import CacheKeys
    from app.core.redis import redis_cache

    await redis_cache.delete_by_prefix(CacheKeys.user_sessions_prefix(current_user.id))
    await redis_cache.delete_by_prefix(CacheKeys.session_details_prefix(session_id))
    return {"message": "Session deleted successfully"}


@router.patch("/sessions/{session_id}/read", status_code=200)
@router.post("/sessions/{session_id}/read", status_code=200)
async def mark_session_read(
    session_id: uuid.UUID,
    current_user: UserDB = Depends(get_current_user),
    db: AsyncClient = Depends(get_db),
):
    repo = ChatRepository(db)
    session, _ = await repo.get_session(session_id, current_user.id, limit=1)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    await repo.mark_session_read(session_id)

    from app.core.cache_keys import CacheKeys
    from app.core.redis import redis_cache

    await redis_cache.delete_by_prefix(CacheKeys.user_sessions_prefix(current_user.id))
    await redis_cache.delete_by_prefix(CacheKeys.session_details_prefix(session_id))


@router.get("/generations/{generation_id}", status_code=200)
async def get_generation(
    generation_id: uuid.UUID,
    response: Response,
    current_user: UserDB = Depends(get_current_user),
    db: AsyncClient = Depends(get_db),
):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"

    from app.repositories.generations import GenerationRepository

    repo = GenerationRepository(db)
    generation = await repo.get_by_id(generation_id)
    if not generation:
        raise HTTPException(status_code=404, detail="Generation not found")
    if str(generation.user_id) != str(current_user.id):
        raise HTTPException(status_code=403, detail="Not authorized")
    return generation.model_dump(mode="json")
