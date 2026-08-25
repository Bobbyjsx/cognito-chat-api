import json


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_signup(client):
    response = client.post("/auth/signup", json={"email": "newuser@example.com", "password": "newpassword123"})
    assert response.status_code == 201
    assert response.json()["email"] == "newuser@example.com"


def test_login(client):
    client.post("/auth/signup", json={"email": "testuser@example.com", "password": "securepassword123"})

    response = client.post("/auth/login", json={"email": "testuser@example.com", "password": "securepassword123"})
    assert response.status_code == 200
    assert "access_token" in response.json()
    assert response.json()["token_type"] == "bearer"


def test_get_me(client):
    client.post("/auth/signup", json={"email": "testuser@example.com", "password": "securepassword123"})

    login_response = client.post("/auth/login", json={"email": "testuser@example.com", "password": "securepassword123"})
    token = login_response.json()["access_token"]

    response = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json()["email"] == "testuser@example.com"


def test_chat(client, mock_agent):
    client.post("/auth/signup", json={"email": "testuser@example.com", "password": "securepassword123"})

    login_response = client.post("/auth/login", json={"email": "testuser@example.com", "password": "securepassword123"})
    token = login_response.json()["access_token"]

    response = client.post("/agent/chat", headers={"Authorization": f"Bearer {token}"}, json={"message": "Hello!"})

    assert response.status_code == 200
    assert "session_id" in response.json()
    assert response.json()["response"] == "Hello from mocked agent!"


def test_stream_unauthorized_returns_401(client):
    """Verify that calling stream endpoint without valid token returns HTTP 401 Unauthorized."""
    resp = client.post(
        "/agent/chat/stream",
        json={"message": "hello"},
        headers={"Accept": "text/event-stream"},
    )
    assert resp.status_code == 401


def test_stream_done_event_includes_model_and_reasoning(client, mock_agent):
    """Verify that stream completion done event returns model and reasoning metadata."""
    client.post("/auth/signup", json={"email": "streamuser@example.com", "password": "securepassword123"})
    login_resp = client.post("/auth/login", json={"email": "streamuser@example.com", "password": "securepassword123"})
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}", "Accept": "text/event-stream"}

    resp = client.post(
        "/agent/chat/stream",
        headers=headers,
        json={"message": "hello", "model": "gemini-3.6-flash", "reasoning": "extended"},
    )
    assert resp.status_code == 200
    content = resp.text
    assert "event: done" in content

    # Parse the data line immediately following `event: done` (not the earlier session event)
    lines = content.splitlines()
    done_data = None
    for i, line in enumerate(lines):
        if line.strip() == "event: done" and i + 1 < len(lines) and lines[i + 1].startswith("data:"):
            done_data = json.loads(lines[i + 1].removeprefix("data:").strip())
            break
    assert done_data is not None, "done event missing data payload"
    assert done_data.get("model") == "gemini-3.6-flash"
    assert done_data.get("reasoning") == "extended"
    assert "session_id" in done_data


def test_refresh_token(client):
    client.post("/auth/signup", json={"email": "refreshtest@example.com", "password": "securepassword123"})

    login_response = client.post(
        "/auth/login", json={"email": "refreshtest@example.com", "password": "securepassword123"}
    )
    assert login_response.status_code == 200

    tokens = login_response.json()
    refresh_token = tokens["refresh_token"]

    refresh_response = client.post("/auth/refresh", json={"refresh_token": refresh_token})

    assert refresh_response.status_code == 200
    new_tokens = refresh_response.json()
    assert "access_token" in new_tokens
    assert "refresh_token" in new_tokens

    me_response = client.get("/auth/me", headers={"Authorization": f"Bearer {new_tokens['access_token']}"})
    assert me_response.status_code == 200
    assert me_response.json()["email"] == "refreshtest@example.com"


def test_sessions(client, mock_agent):
    client.post("/auth/signup", json={"email": "sessionuser@example.com", "password": "securepassword123"})

    login_response = client.post(
        "/auth/login", json={"email": "sessionuser@example.com", "password": "securepassword123"}
    )
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    chat_resp = client.post("/agent/chat", headers=headers, json={"message": "Hello session!"})
    assert chat_resp.status_code == 200
    session_id = chat_resp.json()["session_id"]

    list_resp = client.get("/agent/sessions", headers=headers)
    assert list_resp.status_code == 200
    sessions_data = list_resp.json()
    sessions = sessions_data.get("items", sessions_data) if isinstance(sessions_data, dict) else sessions_data
    assert len(sessions) > 0

    my_session = next(s for s in sessions if s["id"] == session_id)
    assert "messages" not in my_session
    assert my_session["last_message_content"] == "Hello from mocked agent!"
    assert my_session["last_message_role"] == "agent"
    assert my_session["read_status"] == "not read"

    get_resp = client.get(f"/agent/sessions/{session_id}", headers=headers)
    assert get_resp.status_code == 200
    session_detail = get_resp.json()
    assert "messages" in session_detail
    msgs = (
        session_detail["messages"]["items"]
        if isinstance(session_detail["messages"], dict)
        else session_detail["messages"]
    )
    assert len(msgs) == 2
    assert (
        session_detail["session"]["read_status"] == "read"
        if "session" in session_detail
        else session_detail["read_status"] == "read"
    )

    list_resp2 = client.get("/agent/sessions", headers=headers)
    sessions_data2 = list_resp2.json()
    sessions2 = sessions_data2.get("items", sessions_data2) if isinstance(sessions_data2, dict) else sessions_data2
    my_session2 = next(s for s in sessions2 if s["id"] == session_id)
    assert my_session2["read_status"] == "read"

    client.post("/agent/chat", headers=headers, json={"message": "Second message"}, params={"session_id": session_id})
    list_resp3 = client.get("/agent/sessions", headers=headers)
    sessions_data3 = list_resp3.json()
    sessions3 = sessions_data3.get("items", sessions_data3) if isinstance(sessions_data3, dict) else sessions_data3
    my_session3 = next(s for s in sessions3 if s["id"] == session_id)
    assert my_session3["read_status"] == "not read"

    read_resp = client.post(f"/agent/sessions/{session_id}/read", headers=headers)
    assert read_resp.status_code == 200

    list_resp4 = client.get("/agent/sessions", headers=headers)
    sessions_data4 = list_resp4.json()
    sessions4 = sessions_data4.get("items", sessions_data4) if isinstance(sessions_data4, dict) else sessions_data4
    my_session4 = next(s for s in sessions4 if s["id"] == session_id)
    assert my_session4["read_status"] == "read"
