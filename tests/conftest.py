import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Set environment variables for the Firestore emulator
os.environ["FIRESTORE_EMULATOR_HOST"] = os.environ.get("FIRESTORE_EMULATOR_HOST", "127.0.0.1:8080")
os.environ["GOOGLE_CLOUD_PROJECT"] = os.environ.get("GOOGLE_CLOUD_PROJECT", "test-project")
os.environ["FIRESTORE_DATABASE"] = os.environ.get("FIRESTORE_DATABASE", "(default)")

# Attachments use the local storage backend in tests
os.environ["STORAGE_BACKEND"] = "local"
os.environ["LOCAL_STORAGE_DIR"] = os.environ.get("TEST_STORAGE_DIR", "/tmp/cognito-test-storage")

# Bypass password hashing in tests to prevent bcrypt bugs with passlib
patch("app.core.security.verify_password", return_value=True).start()
patch("app.core.security.get_password_hash", return_value="hashed").start()

# Bypass real firebase_admin initialization so it doesn't try to use real credentials
patch("firebase_admin.initialize_app").start()
patch("firebase_admin._apps", {"[DEFAULT]": True}).start()


# Mock Redis to prevent touching a real/live database in tests
class MockRedisClient:
    def __init__(self):
        self.store = {}

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.store[key] = value

    async def delete(self, *keys):
        for k in keys:
            self.store.pop(k, None)

    async def scan(self, cursor=0, match=None, count=None):
        if match:
            prefix = match.replace("*", "")
            keys = [k for k in self.store if k.startswith(prefix)]
        else:
            keys = list(self.store)
        return 0, keys

    async def flushdb(self):
        self.store.clear()

    async def ping(self):
        return True

    async def aclose(self):
        pass


patch("redis.asyncio.from_url", return_value=MockRedisClient()).start()


class MockApp:
    project_id = "test-project"


patch("firebase_admin.get_app", return_value=MockApp()).start()

from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(autouse=True)
def clear_database():
    """Clear the Firestore emulator database, re-seed app_config, and clear Redis between tests."""
    from app.core.redis import redis_cache

    if redis_cache.redis_client:
        if hasattr(redis_cache.redis_client, "store"):
            redis_cache.redis_client.store.clear()
        else:
            import asyncio

            try:
                loop = asyncio.get_running_loop()
                loop.create_task(redis_cache.redis_client.flushdb())
            except RuntimeError:
                asyncio.run(redis_cache.redis_client.flushdb())
    try:
        import urllib.request

        emulator_host = os.environ.get("FIRESTORE_EMULATOR_HOST", "127.0.0.1:8080")
        # Emulator REST API for wiping the database
        req = urllib.request.Request(
            f"http://{emulator_host}/emulator/v1/projects/test-project/databases/(default)/documents",
            method="DELETE",
        )
        urllib.request.urlopen(req, timeout=2.0)
    except Exception:
        emulator_host = os.environ.get("FIRESTORE_EMULATOR_HOST", "127.0.0.1:8080")
        print(f"Warning: Could not connect to Firestore emulator at {emulator_host}. Is it running?")

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
    # Mock the google-genai Client (used by the GeminiProvider) so we don't hit the real Gemini API
    with patch("app.providers.gemini.genai.Client") as MockClientClass:
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
                self.candidates = []

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
        mock_client_instance.aio.files.upload = AsyncMock(return_value=MagicMock(uri="files/mock-upload"))

        yield mock_client_instance


# Mock Cloud Tasks so tests do not actually hit GCP
import contextlib

with contextlib.suppress(Exception):
    patch("google.cloud.tasks_v2.CloudTasksClient").start()
