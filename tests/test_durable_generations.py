import datetime
import uuid

import jwt
import pytest
from fastapi.testclient import TestClient
from google.cloud import firestore

from app.models.chats import GenerationDB, GenerationStatus


@pytest.fixture
def test_user_and_auth(client: TestClient):
    """Creates a test user and returns auth headers."""
    email = f"test_{uuid.uuid4().hex}@example.com"
    client.post("/auth/signup", json={"email": email, "password": "password123"})
    resp = client.post("/auth/login", json={"email": email, "password": "password123"})
    token = resp.json()["access_token"]

    db = firestore.Client(project="test-project")
    payload = jwt.decode(token, options={"verify_signature": False})
    user_id = payload["sub"]

    return {"headers": {"Authorization": f"Bearer {token}"}, "user_id": user_id, "db": db}


def test_abandoned_stream_creates_durable_record(client: TestClient, test_user_and_auth: dict, mock_agent):
    """Test that stream abandonment creates a durable generation record and worker completes it."""
    db: firestore.Client = test_user_and_auth["db"]
    user_id = test_user_and_auth["user_id"]

    session_id = str(uuid.uuid4())
    db.collection("sessions").document(session_id).set(
        {
            "id": session_id,
            "user_id": user_id,
            "title": "Test session",
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "is_deleted": False,
            "read_status": "read",
        }
    )

    gen_id = str(uuid.uuid4())
    generation = GenerationDB(
        id=gen_id,
        user_id=user_id,
        session_id=session_id,
        status=GenerationStatus.QUEUED,
        prompt="Hello background worker",
        requested_model="gemini-2.5-flash",
        resolved_model="gemini-2.5-flash",
    )
    db.collection("generations").document(gen_id).set(generation.model_dump(mode="json"))

    response = client.post(
        f"/tasks/generations/{gen_id}",
        headers={"X-CloudTasks-QueueName": "cognito-generations"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("claimed", "success", "sent")

    gen_doc = db.collection("generations").document(gen_id).get()
    assert gen_doc.exists
    gen_data = gen_doc.to_dict()
    assert gen_data["status"] == GenerationStatus.COMPLETED.value
    assert gen_data.get("message_id") is not None


def test_background_worker_claims_and_executes(client: TestClient, test_user_and_auth: dict, mock_agent):
    """Test that the worker can claim a stale generation and execute it."""
    db: firestore.Client = test_user_and_auth["db"]
    user_id = test_user_and_auth["user_id"]

    session_id = str(uuid.uuid4())
    db.collection("sessions").document(session_id).set(
        {
            "id": session_id,
            "user_id": user_id,
            "title": "Test session",
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "is_deleted": False,
            "read_status": "read",
        }
    )

    msg_id = str(uuid.uuid4())
    db.collection("sessions").document(session_id).collection("messages").document(msg_id).set(
        {
            "id": msg_id,
            "session_id": session_id,
            "role": "user",
            "content": "Hello background worker",
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
    )

    generation_id = str(uuid.uuid4())
    stale_time = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=20)).isoformat()
    generation = GenerationDB(
        id=generation_id,
        user_id=user_id,
        session_id=session_id,
        status=GenerationStatus.RUNNING_LIVE,
        requested_model="gemini-2.5-flash",
        resolved_model="gemini-2.5-flash",
        prompt="Hello background worker",
        created_at=stale_time,
        updated_at=stale_time,
    )
    db.collection("generations").document(generation_id).set(generation.model_dump(mode="json"))

    payload = {"generation_id": generation_id, "attempt_number": 1}

    response = client.post(
        f"/tasks/generations/{generation_id}",
        json=payload,
        headers={"X-CloudTasks-QueueName": "cognito-generations"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("claimed", "success", "sent")

    gen_doc = db.collection("generations").document(generation_id).get()
    assert gen_doc.exists
    gen_data = gen_doc.to_dict()
    assert gen_data["status"] == GenerationStatus.COMPLETED.value
    assert gen_data.get("message_id") is not None


def test_local_worker_provider_returns_no_dispatcher():
    """Test that when worker_provider is 'local', get_tasks_dispatcher returns None."""
    from unittest.mock import MagicMock

    from app.api.dependencies import get_tasks_dispatcher
    from app.core.config import settings

    orig_provider = settings.worker_provider
    try:
        settings.worker_provider = "local"
        mock_request = MagicMock()
        mock_request.app.state.tasks_dispatcher = None

        dispatcher = get_tasks_dispatcher(mock_request)
        assert dispatcher is None
    finally:
        settings.worker_provider = orig_provider
