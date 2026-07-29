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

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


@pytest.fixture(autouse=True)
def clear_database():
    """Clear the Firestore emulator database between each test."""
    try:
        # Emulator REST API for wiping the database
        httpx.delete("http://127.0.0.1:8080/emulator/v1/projects/test-project/databases/(default)/documents")
    except httpx.RequestError:
        print("Warning: Could not connect to Firestore emulator at 127.0.0.1:8080. Is it running?")
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

        class MockChunk:
            def __init__(self, text):
                self.text = text
                self.usage_metadata = MagicMock(total_token_count=10)

        class MockResponse:
            def __init__(self):
                self.text = "Hello from mocked agent!"
                self.usage_metadata = MagicMock(total_token_count=10)

        async def mock_generate_content(*args, **kwargs):
            return MockResponse()

        async def mock_generate_content_stream(*args, **kwargs):
            for t in ["Hello ", "from ", "mocked ", "agent!"]:
                yield MockChunk(t)

        mock_client_instance.aio.models.generate_content = AsyncMock(side_effect=mock_generate_content)
        mock_client_instance.aio.models.generate_content_stream = MagicMock(side_effect=mock_generate_content_stream)

        yield mock_client_instance
