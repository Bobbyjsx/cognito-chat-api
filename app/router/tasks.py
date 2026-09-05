from fastapi import APIRouter, Depends, Request, status
from google.cloud.firestore_v1.async_client import AsyncClient

from app.api.dependencies import verify_cloud_tasks_caller
from app.database import get_db
from app.repositories.chats import ChatRepository
from app.repositories.config import ConfigRepository
from app.repositories.generations import GenerationRepository
from app.repositories.users import UserRepository
from app.router.chats import get_agent_service
from app.schemas.task import (
    GenerationTaskPayload,
    TaskExecutionResponse,
    TitleTaskExecutionResponse,
    TitleTaskPayload,
)
from app.services.generation_worker import GenerationWorkerService

router = APIRouter(prefix="/tasks", tags=["Tasks Worker"])


def get_generation_worker_service(
    db: AsyncClient = Depends(get_db),
    agent_service=Depends(get_agent_service),
) -> GenerationWorkerService:
    return GenerationWorkerService(
        generation_repo=GenerationRepository(db),
        chat_repo=ChatRepository(db),
        user_repo=UserRepository(db),
        config_repo=ConfigRepository(db),
        agent_service=agent_service,
    )


@router.post(
    "/generations/{generation_id}",
    response_model=TaskExecutionResponse,
    status_code=status.HTTP_200_OK,
    summary="Cloud Tasks asynchronous generation worker",
)
async def execute_generation_task(
    generation_id: str,
    request: Request,
    is_authorized: bool = Depends(verify_cloud_tasks_caller),
    worker: GenerationWorkerService = Depends(get_generation_worker_service),
) -> TaskExecutionResponse:
    """
    Asynchronously executes an AI generation scheduled by Cloud Tasks.
    Idempotent and safe to execute multiple times.
    """
    # Cloud Tasks might pass body or might just call URL.
    # The payload is usually in body, but we take generation_id from path and build payload.
    # To support attempt_number we could parse headers.
    attempt_number = int(request.headers.get("X-CloudTasks-TaskRetryCount", "0")) + 1
    payload = GenerationTaskPayload(generation_id=generation_id, attempt_number=attempt_number)
    return await worker.execute_task(payload)


@router.post(
    "/titles/{session_id}",
    response_model=TitleTaskExecutionResponse,
    status_code=status.HTTP_200_OK,
    summary="Cloud Tasks asynchronous title worker",
)
async def execute_title_task(
    session_id: str,
    payload: TitleTaskPayload,
    request: Request,
    is_authorized: bool = Depends(verify_cloud_tasks_caller),
    worker: GenerationWorkerService = Depends(get_generation_worker_service),
) -> TitleTaskExecutionResponse:
    """
    Asynchronously generates a concise AI session title scheduled by Cloud Tasks.
    """
    attempt_number = int(request.headers.get("X-CloudTasks-TaskRetryCount", "0")) + 1
    payload.attempt_number = attempt_number
    return await worker.execute_title_task(payload)
