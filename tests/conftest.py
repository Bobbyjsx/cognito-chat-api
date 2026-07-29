import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

# Set environment variables for the Firestore emulator
os.environ["FIRESTORE_EMULATOR_HOST"] = "127.0.0.1:8080"
os.environ["GOOGLE_CLOUD_PROJECT"] = "test-project"
os.environ["FIRESTORE_DATABASE"] = "(default)"

# Bypass password hashing in tests to prevent bcrypt bugs with passlib
patch("app.core.security.verify_password", return_value=True).start()
patch("app.core.security.get_password_hash", return_value="hashed").start()

# Bypass real firebase_admin initialization so it doesn't try to use real credentials
patch("firebase_admin.initialize_app").start()
patch("firebase_admin._apps", {"[DEFAULT]": True}).start()


class MockApp:
    project_id = "test-project"


patch("firebase_admin.get_app", return_value=MockApp()).start()

from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(autouse=True)
def clear_database():
    """Clear the Firestore emulator database and re-seed app_config between tests."""
    try:
        # Emulator REST API for wiping the database
        httpx.delete(
            "http://127.0.0.1:8080/emulator/v1/projects/test-project/databases/(default)/documents", timeout=2.0
        )
    except httpx.RequestError:
        print("Warning: Could not connect to Firestore emulator at 127.0.0.1:8080. Is it running?")

    # Chat/config paths require configs/app_config; seed defaults after wipe.
    try:
        from google.cloud import firestore

        from app.models.config import AppConfigDB

        db = firestore.Client(project="test-project")
        data = AppConfigDB().model_dump(mode="json")
        db.collection("configs").document("app_config").set(data)
    except Exception as exc:
        print(f"Warning: Could not seed app_config: {exc}")
    yield


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def mock_agent():
    # Mock the google-genai Client so we don't hit the real Gemini API
    with patch("app.services.chats.genai.Client") as MockClientClass:
        mock_client_instance = MockClientClass.return_value

        class MockPart:
            def __init__(self, text=None, thought=False, function_call=None):
                self.text = text
                self.thought = thought
                self.function_call = function_call
                self.executable_code = None
                self.code_execution_result = None
                self.function_response = None

        class MockChunk:
            def __init__(self, text, thought=False):
                self.text = text
                part = MockPart(text=text, thought=thought)
                content = MagicMock(parts=[part])
                candidate = MagicMock(content=content)
                self.candidates = [candidate]
                self.usage_metadata = MagicMock(total_token_count=10)

        class MockResponse:
            def __init__(self):
                self.text = "Hello from mocked agent!"
                self.usage_metadata = MagicMock(total_token_count=10)

        async def mock_generate_content(*args, **kwargs):
            return MockResponse()

        async def mock_generate_content_stream(*args, **kwargs):
            # Service awaits this call, then async-iterates the result.
            async def _stream():
                for t in ["Hello ", "from ", "mocked ", "agent!"]:
                    yield MockChunk(t)

            return _stream()

        mock_client_instance.aio.models.generate_content = AsyncMock(side_effect=mock_generate_content)
        mock_client_instance.aio.models.generate_content_stream = AsyncMock(side_effect=mock_generate_content_stream)

        yield mock_client_instance
