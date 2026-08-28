import asyncio
import logging
from typing import Any

from app.core.config import settings
from app.schemas.task import GenerationTaskPayload

logger = logging.getLogger(__name__)


class CloudTasksDispatcher:
    """Dispatches asynchronous generation tasks to Google Cloud Tasks."""

    def __init__(
        self,
        project: str | None = None,
        location: str | None = None,
        queue: str | None = None,
        worker_url: str | None = None,
        service_account_email: str | None = None,
    ) -> None:
        self.project = project or settings.cloud_tasks_project
        self.location = location or settings.cloud_tasks_location
        self.queue = queue or settings.cloud_tasks_queue
        self.worker_url = worker_url or settings.cloud_tasks_worker_url
        self.service_account_email = service_account_email or settings.cloud_tasks_service_account_email
        self._client: Any = None

    def _get_client(self):
        if self._client is None:
            try:
                from google.cloud import tasks_v2

                self._client = tasks_v2.CloudTasksClient()
            except Exception as exc:
                logger.warning("Could not initialize CloudTasksClient: %s", exc)
                self._client = None
        return self._client

    async def enqueue_generation_task(self, payload: GenerationTaskPayload) -> str:
        """
        Enqueues an asynchronous generation execution task.
        Returns the created task ID or a simulated task ID in dev/mock mode.
        """
        client = self._get_client()
        if not client:
            raise RuntimeError("CloudTasks client is not available")

        from google.cloud import tasks_v2

        parent = client.queue_path(self.project, self.location, self.queue)

        body_bytes = payload.model_dump_json().encode("utf-8")

        task: dict[str, Any] = {
            "http_request": {
                "http_method": tasks_v2.HttpMethod.POST,
                "url": f"{self.worker_url.rstrip('/')}/{payload.generation_id}",
                "headers": {
                    "Content-Type": "application/json",
                    "User-Agent": "cognito-chat-api-cloud-tasks/1.0",
                },
                "body": body_bytes,
            }
        }

        # Add OIDC token for Cloud Run worker authentication if service account is configured
        if self.service_account_email:
            task["http_request"]["oidc_token"] = {
                "service_account_email": self.service_account_email,
                "audience": self.worker_url,
            }

        # Offload sync SDK call to worker thread to avoid blocking asyncio event loop
        def _sync_create_task():
            return client.create_task(
                tasks_v2.CreateTaskRequest(
                    parent=parent,
                    task=task,
                )
            )

        try:
            created_task = await asyncio.to_thread(_sync_create_task)
            task_name = created_task.name
            logger.info("Successfully enqueued Cloud Task: %s for generation %s", task_name, payload.generation_id)
            return task_name
        except Exception:
            logger.exception("Failed to enqueue Cloud Task for generation %s", payload.generation_id)
            raise
