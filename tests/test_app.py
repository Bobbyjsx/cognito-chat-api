def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_signup(client):
    response = client.post("/auth/signup", json={"email": "newuser@example.com", "password": "newpassword123"})
    # Even though it's a mock, it will simulate a successful creation
    print("SIGNUP RESPONSE:", response.json())
    assert response.status_code == 201
    assert response.json()["email"] == "newuser@example.com"


def test_login(client):
    # Create the user first
    client.post("/auth/signup", json={"email": "testuser@example.com", "password": "securepassword123"})

    response = client.post("/auth/login", json={"email": "testuser@example.com", "password": "securepassword123"})
    assert response.status_code == 200
    assert "access_token" in response.json()
    assert response.json()["token_type"] == "bearer"


def test_get_me(client):
    # Create the user first
    client.post("/auth/signup", json={"email": "testuser@example.com", "password": "securepassword123"})

    # First login to get a token
    login_response = client.post("/auth/login", json={"email": "testuser@example.com", "password": "securepassword123"})
    token = login_response.json()["access_token"]

    response = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json()["email"] == "testuser@example.com"


def test_chat(client, mock_agent):
    # Create the user first
    client.post("/auth/signup", json={"email": "testuser@example.com", "password": "securepassword123"})

    # First login to get a token
    login_response = client.post("/auth/login", json={"email": "testuser@example.com", "password": "securepassword123"})
    token = login_response.json()["access_token"]

    response = client.post("/agent/chat", headers={"Authorization": f"Bearer {token}"}, json={"message": "Hello!"})

    assert response.status_code == 200
    assert "session_id" in response.json()
    assert response.json()["response"] == "Hello from mocked agent!"


def test_refresh_token(client):
    # Create the user
    client.post("/auth/signup", json={"email": "refreshtest@example.com", "password": "securepassword123"})

    # Login to get tokens
    login_response = client.post(
        "/auth/login", json={"email": "refreshtest@example.com", "password": "securepassword123"}
    )
    assert login_response.status_code == 200

    tokens = login_response.json()
    # old_access_token = tokens["access_token"]
    refresh_token = tokens["refresh_token"]

    # Refresh the token
    refresh_response = client.post("/auth/refresh", json={"refresh_token": refresh_token})

    assert refresh_response.status_code == 200
    new_tokens = refresh_response.json()
    assert "access_token" in new_tokens
    assert "refresh_token" in new_tokens

    # The new access token should be valid (it might be identical to the old one if generated in the same second)
    # Verify the new token works
    me_response = client.get("/auth/me", headers={"Authorization": f"Bearer {new_tokens['access_token']}"})
    assert me_response.status_code == 200
    assert me_response.json()["email"] == "refreshtest@example.com"


def test_sessions(client, mock_agent):
    # Create the user first
    client.post("/auth/signup", json={"email": "sessionuser@example.com", "password": "securepassword123"})

    # Login to get a token
    login_response = client.post(
        "/auth/login", json={"email": "sessionuser@example.com", "password": "securepassword123"}
    )
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 1. Chat to create a session
    chat_resp = client.post("/agent/chat", headers=headers, json={"message": "Hello session!"})
    assert chat_resp.status_code == 200
    session_id = chat_resp.json()["session_id"]

    # 2. List sessions and check metadata
    list_resp = client.get("/agent/sessions", headers=headers)
    assert list_resp.status_code == 200
    sessions = list_resp.json()
    assert len(sessions) > 0

    # Find our session
    my_session = next(s for s in sessions if s["id"] == session_id)
    assert "messages" not in my_session
    assert my_session["last_message_content"] == "Hello from mocked agent!"
    assert my_session["last_message_role"] == "agent"
    assert my_session["read_status"] == "not read"

    # 3. Open session (marks as read)
    get_resp = client.get(f"/agent/sessions/{session_id}", headers=headers)
    assert get_resp.status_code == 200
    session_detail = get_resp.json()
    assert "messages" in session_detail
    assert len(session_detail["messages"]) == 2
    assert session_detail["read_status"] == "read"

    # 4. List sessions again, should be read
    list_resp2 = client.get("/agent/sessions", headers=headers)
    sessions2 = list_resp2.json()
    my_session2 = next(s for s in sessions2 if s["id"] == session_id)
    assert my_session2["read_status"] == "read"

    # 5. Send another message (agent replies, marks as unread)
    client.post("/agent/chat", headers=headers, json={"message": "Second message"}, params={"session_id": session_id})
    list_resp3 = client.get("/agent/sessions", headers=headers)
    my_session3 = next(s for s in list_resp3.json() if s["id"] == session_id)
    assert my_session3["read_status"] == "not read"

    # 6. Manually mark as read via endpoint
    read_resp = client.post(f"/agent/sessions/{session_id}/read", headers=headers)
    assert read_resp.status_code == 200

    list_resp4 = client.get("/agent/sessions", headers=headers)
    my_session4 = next(s for s in list_resp4.json() if s["id"] == session_id)
    assert my_session4["read_status"] == "read"
