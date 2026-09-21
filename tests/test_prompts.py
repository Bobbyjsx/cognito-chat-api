from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_current_user
from app.database import get_db
from app.main import app
from app.models.users import UserDB


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_user():
    return UserDB(
        id="user-1234",
        email="test@cognito.ai",
        hashed_password="hash",
    )


def test_list_prompts_forbidden_for_free_user(client, mock_user):
    app.dependency_overrides[get_current_user] = lambda: mock_user
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db

    try:
        with patch("app.router.prompts.lookup_active_tier", new_callable=AsyncMock) as mock_tier:
            mock_tier.return_value = "free"
            response = client.get("/prompts")
            assert response.status_code == 403
            assert "Premium" in response.json()["detail"]
    finally:
        app.dependency_overrides.clear()


def test_list_prompts_forbidden_for_go_user(client, mock_user):
    app.dependency_overrides[get_current_user] = lambda: mock_user
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db

    try:
        with patch("app.router.prompts.lookup_active_tier", new_callable=AsyncMock) as mock_tier:
            mock_tier.return_value = "go"
            response = client.get("/prompts")
            assert response.status_code == 403
            assert "Premium" in response.json()["detail"]
    finally:
        app.dependency_overrides.clear()


def test_create_prompt_forbidden_for_non_premium(client, mock_user):
    app.dependency_overrides[get_current_user] = lambda: mock_user
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db

    try:
        with patch("app.router.prompts.lookup_active_tier", new_callable=AsyncMock) as mock_tier:
            mock_tier.return_value = "go"
            response = client.post(
                "/prompts",
                json={
                    "title": "My Custom Prompt",
                    "description": "Description",
                    "prompt": "Custom prompt text",
                },
            )
            assert response.status_code == 403
    finally:
        app.dependency_overrides.clear()


def test_list_prompts_allowed_for_premium_user(client, mock_user):
    app.dependency_overrides[get_current_user] = lambda: mock_user
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db

    try:
        with (
            patch("app.router.prompts.lookup_active_tier", new_callable=AsyncMock) as mock_tier,
            patch("app.router.prompts.PromptRepository") as mock_repo_cls,
        ):
            mock_tier.return_value = "premium"
            mock_repo = MagicMock()
            mock_repo.get_user_prompts = AsyncMock(return_value=[])
            mock_repo_cls.return_value = mock_repo

            response = client.get("/prompts")
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
            assert len(data) > 0
            # Contains built-in prompts
            assert any(p["id"] == "code-refactor" for p in data)
    finally:
        app.dependency_overrides.clear()
