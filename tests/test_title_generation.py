import datetime
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from google.cloud import firestore

from app.repositories.chats import ChatRepository
from app.repositories.config import ConfigRepository
from app.repositories.generations import GenerationRepository
from app.repositories.users import UserRepository
from app.schemas.task import TitleTaskPayload
from app.services.chats import AgentService
from app.services.generation_worker import GenerationWorkerService


def test_sanitize_title_prompt():
    """Verify that sanitize_title_prompt strips markdown, urls, and caps chars strictly at 160."""
    long_code = (
        "Can you fix this code?\n```python\n" + ("x = 1\n" * 50) + "```\nCheck this url: https://example.com/api"
    )
    sanitized = AgentService.sanitize_title_prompt(long_code)
    assert "x = 1" not in sanitized
    assert "https://" not in sanitized
    assert len(sanitized) <= 160
    assert sanitized.startswith("Can you fix this code?")


def test_evaluate_title_strategy_local_patterns():
    """Verify high-confidence template prompts resolve locally with needs_ai_worker=False."""
    # Geography assignment
    title, needs_worker = AgentService._evaluate_title_strategy(
        "Can you help me do my geography assignment about tectonic plates?"
    )
    assert title == "Assistance with geography assignment"
    assert needs_worker is False

    # Python script
    title, needs_worker = AgentService._evaluate_title_strategy(
        "Please write a python script to scrape product prices from ebay"
    )
    assert "script" in title.lower()
    assert needs_worker is False

    # Debugging
    title, needs_worker = AgentService._evaluate_title_strategy("Debug this KeyError in my FastAPI endpoint")
    assert title.startswith("Debugging")
    assert needs_worker is False

    # Greeting
    title, needs_worker = AgentService._evaluate_title_strategy("Hello there")
    assert title == "New Chat"
    assert needs_worker is False


def test_evaluate_title_strategy_ai_fallback():
    """Verify non-pattern complex prompts trigger needs_ai_worker=True."""
    prompt = (
        "We are rearchitecting our distributed message queue system because consumer lag is exceeding SLA. "
        "Can we review our partition sizing strategy?"
    )
    title, needs_worker = AgentService._evaluate_title_strategy(prompt)
    assert needs_worker is True
    assert title  # has immediate fallback title


from google.cloud.firestore_v1.async_client import AsyncClient


@pytest.mark.asyncio
async def test_resolve_session_stores_title_in_db():
    """Verify _resolve_session always persists title in Firestore."""
    db = AsyncClient(project="test-project")
    chat_repo = ChatRepository(db)
    agent = AgentService(
        chat_repo=chat_repo,
        user_repo=MagicMock(),
        config_repo=MagicMock(),
        attachment_service=MagicMock(),
        provider=MagicMock(),
    )

    user_id = str(uuid.uuid4())
    mock_user = MagicMock(id=user_id)

    # 1. New session
    _session, session_id, title, needs_worker = await agent._resolve_session(
        user=mock_user,
        session_id=None,
        message_text="Can you help me do my geography assignment about tectonic plates?",
    )
    assert title == "Assistance with geography assignment"
    assert needs_worker is False

    # Verify Firestore document has the title stored
    doc = await db.collection("sessions").document(str(session_id)).get()
    assert doc.exists
    data = doc.to_dict()
    assert data["title"] == "Assistance with geography assignment"
    assert data["user_id"] == user_id


@pytest.mark.asyncio
async def test_worker_executes_title_task_and_updates_db():
    """Verify GenerationWorkerService.execute_title_task generates AI title and updates DB."""
    db = AsyncClient(project="test-project")
    chat_repo = ChatRepository(db)
    gen_repo = GenerationRepository(db)
    user_repo = UserRepository(db)
    config_repo = ConfigRepository(db)

    user_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())

    # Pre-create session in DB with initial placeholder title
    await (
        db.collection("sessions")
        .document(session_id)
        .set(
            {
                "id": session_id,
                "user_id": user_id,
                "title": "Temporary Title",
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "is_deleted": False,
                "read_status": "read",
            }
        )
    )

    mock_agent = MagicMock()
    mock_agent.generate_ai_title = AsyncMock(return_value="Distributed Queue Partitioning")

    worker = GenerationWorkerService(
        generation_repo=gen_repo,
        chat_repo=chat_repo,
        user_repo=user_repo,
        config_repo=config_repo,
        agent_service=mock_agent,
    )

    payload = TitleTaskPayload(
        session_id=session_id,
        user_id=user_id,
        prompt="Review partition sizing strategy",
    )

    res = await worker.execute_title_task(payload)
    assert res.status == "completed"
    assert res.title == "Distributed Queue Partitioning"

    # Verify Firestore document was updated with the AI title
    doc = await db.collection("sessions").document(session_id).get()
    assert doc.exists
    data = doc.to_dict()
    assert data["title"] == "Distributed Queue Partitioning"


def test_tasks_titles_endpoint(client: TestClient):
    """Test POST /tasks/titles/{session_id} invokes worker and returns 200."""
    db = firestore.Client(project="test-project")
    user_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())

    db.collection("sessions").document(session_id).set(
        {
            "id": session_id,
            "user_id": user_id,
            "title": "Initial Title",
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "is_deleted": False,
            "read_status": "read",
        }
    )

    with patch("app.services.chats.AgentService.generate_ai_title", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = "Refined Kubernetes Guide"

        payload = {
            "session_id": session_id,
            "user_id": user_id,
            "prompt": "Fix Kubernetes pod evictions",
        }

        response = client.post(
            f"/tasks/titles/{session_id}",
            json=payload,
            headers={"X-CloudTasks-QueueName": "cognito-generations"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["title"] == "Refined Kubernetes Guide"

        doc = db.collection("sessions").document(session_id).get()
        assert doc.exists
        assert doc.to_dict()["title"] == "Refined Kubernetes Guide"
