import uuid

import pytest
from fastapi.testclient import TestClient

from app.models.chats import GenerationStatus
from app.repositories.generations import GenerationRepository


@pytest.fixture
def test_user_and_auth(client: TestClient):
    """Creates a test user and returns auth headers."""
    email = f"test_{uuid.uuid4().hex}@example.com"
    client.post("/auth/signup", json={"email": email, "password": "password123"})
    resp = client.post("/auth/login", json={"email": email, "password": "password123"})
    token = resp.json()["access_token"]

    # We need the user db object
    from app.main import app

    db_client = app.state.db_client

    # Get user id from token
    import jwt

    payload = jwt.decode(token, options={"verify_signature": False})
    user_id = payload["sub"]

    return {"headers": {"Authorization": f"Bearer {token}"}, "user_id": user_id, "db_client": db_client}


@pytest.mark.asyncio
async def test_live_generation_creates_durable_record(client: TestClient, test_user_and_auth: dict, mock_agent):
    """Test that a live generation creates a durable generation record and completes it."""
    payload = {"message": "Hello, generate a background response!"}

    response = client.post("/agent/chat/stream", json=payload, headers=test_user_and_auth["headers"])
    assert response.status_code == 200

    chunks = list(response.iter_text())

    full_output = "".join(chunks)
    assert "event: done" in full_output

    generation_repo = GenerationRepository(test_user_and_auth["db_client"])
    generations = []
    async for doc in generation_repo.collection.stream():
        generations.append(doc.to_dict())

    assert len(generations) > 0
    latest_gen = max(generations, key=lambda x: x["created_at"])

    assert latest_gen["status"] == GenerationStatus.COMPLETED.value
    assert latest_gen["usage_tokens"] > 0
    assert latest_gen["message_id"] is not None


@pytest.mark.asyncio
async def test_background_worker_claims_and_executes(client: TestClient, test_user_and_auth: dict, mock_agent):
    """Test that the worker can claim a generation and execute it."""
    from app.models.chats import GenerationDB, GenerationStatus
    from app.repositories.chats import ChatRepository

    db_client = test_user_and_auth["db_client"]
    generation_repo = GenerationRepository(db_client)
    chat_repo = ChatRepository(db_client)

    user_id = test_user_and_auth["user_id"]
    session = await chat_repo.create_session(user_id, "Test session")
    await chat_repo.add_message(session.id, "user", "Hello background worker")

    generation = GenerationDB(
        user_id=user_id,
        session_id=session.id,
        status=GenerationStatus.RUNNING_LIVE,
        requested_model="gemini-2.5-flash",
        resolved_model="gemini-2.5-flash",
    )

    import datetime

    generation.updated_at = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=20)
    await generation_repo.create(generation)

    payload = {"generation_id": str(generation.id), "attempt_number": 1}

    response = client.post(f"/tasks/generations/{generation.id}", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "sent"

    updated_gen = await generation_repo.get_by_id(generation.id)
    assert updated_gen.status == GenerationStatus.COMPLETED
    assert updated_gen.message_id is not None
