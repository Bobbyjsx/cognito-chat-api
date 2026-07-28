import os
from unittest.mock import MagicMock, patch

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
    # Mock the antigravity Agent so we don't hit the real Gemini API
    with patch("app.services.chats.Agent", autospec=True) as MockAgent:
        instance = MockAgent.return_value.__aenter__.return_value

        class MockResponse:
            def __init__(self):
                self.usage_metadata = MagicMock(total_token_count=10)

            async def __aiter__(self):
                for token in ["Hello ", "from ", "mocked ", "agent!"]:
                    yield token

        instance.chat.return_value = MockResponse()
        yield instance
