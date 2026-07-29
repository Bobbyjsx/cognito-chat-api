from unittest.mock import AsyncMock, patch

from app.models.config import AppConfigDB


def test_get_config_endpoint(client):
    response = client.get("/config")
    assert response.status_code == 200
    data = response.json()
    assert "allowed_text_models" in data
    assert "default_text_model" in data
    assert "allowed_reasoning_levels" in data
    assert "default_reasoning_level" in data
    assert "allowed_image_models" in data
    assert "allowed_video_models" in data
    assert "allowed_tools" in data
    assert "enable_text_generation" in data
    assert "enable_image_generation" in data
    assert "enable_video_generation" in data
    assert "gemini-3.6-flash" in data["allowed_text_models"]


def test_chat_invalid_model_returns_400(client, mock_agent):
    # Signup & login
    client.post("/auth/signup", json={"email": "cfgtest@example.com", "password": "password123"})
    login_resp = client.post("/auth/login", json={"email": "cfgtest@example.com", "password": "password123"})
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Pass model NOT in allowed_text_models (e.g., unauthorized model or image model)
    resp = client.post(
        "/agent/chat",
        headers=headers,
        json={"message": "hello", "model": "invalid-model-999"},
    )
    assert resp.status_code == 400
    assert "not allowed or supported" in resp.json()["detail"]


def test_chat_invalid_reasoning_returns_400(client, mock_agent):
    client.post("/auth/signup", json={"email": "cfgtest2@example.com", "password": "password123"})
    login_resp = client.post("/auth/login", json={"email": "cfgtest2@example.com", "password": "password123"})
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    resp = client.post(
        "/agent/chat",
        headers=headers,
        json={"message": "hello", "reasoning": "ultra-super-high"},
    )
    assert resp.status_code == 400
    assert "Reasoning level 'ultra-super-high' is not allowed" in resp.json()["detail"]


def test_chat_valid_model_and_reasoning_success(client, mock_agent):
    client.post("/auth/signup", json={"email": "cfgtest3@example.com", "password": "password123"})
    login_resp = client.post("/auth/login", json={"email": "cfgtest3@example.com", "password": "password123"})
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    resp = client.post(
        "/agent/chat",
        headers=headers,
        json={"message": "hello", "model": "gemini-3.6-flash", "reasoning": "high"},
    )
    assert resp.status_code == 200
    assert "response" in resp.json()


def test_chat_disabled_text_generation_returns_400(client, mock_agent):
    client.post("/auth/signup", json={"email": "cfgtest4@example.com", "password": "password123"})
    login_resp = client.post("/auth/login", json={"email": "cfgtest4@example.com", "password": "password123"})
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    disabled_config = AppConfigDB(enable_text_generation=False)
    with patch("app.repositories.config.ConfigRepository.get_config", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = disabled_config
        resp = client.post(
            "/agent/chat",
            headers=headers,
            json={"message": "hello"},
        )
        assert resp.status_code == 400
        assert "disabled" in resp.json()["detail"]
