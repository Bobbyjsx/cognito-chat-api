import datetime
import uuid

import jwt
import pytest
from fastapi.testclient import TestClient
from google.cloud import firestore


@pytest.fixture
def test_user(client: TestClient):
    """Creates a primary test user and returns auth headers."""
    email = f"user_{uuid.uuid4().hex}@example.com"
    client.post("/auth/signup", json={"email": email, "password": "password123"})
    resp = client.post("/auth/login", json={"email": email, "password": "password123"})
    token = resp.json()["access_token"]

    db = firestore.Client(project="test-project")
    payload = jwt.decode(token, options={"verify_signature": False})
    user_id = payload["sub"]

    return {"headers": {"Authorization": f"Bearer {token}"}, "user_id": user_id, "db": db}


@pytest.fixture
def secondary_user(client: TestClient):
    """Creates a secondary test user for continuation tests."""
    email = f"importer_{uuid.uuid4().hex}@example.com"
    client.post("/auth/signup", json={"email": email, "password": "password123"})
    resp = client.post("/auth/login", json={"email": email, "password": "password123"})
    token = resp.json()["access_token"]

    payload = jwt.decode(token, options={"verify_signature": False})
    user_id = payload["sub"]

    return {"headers": {"Authorization": f"Bearer {token}"}, "user_id": user_id}


def test_share_session_snapshot_and_point_in_time_freeze(client: TestClient, test_user: dict, secondary_user: dict):
    """Tests that sharing a chat freezes messages at that exact moment, scrubs system roles, and respects show_name."""
    db: firestore.Client = test_user["db"]
    user_id = test_user["user_id"]
    headers = test_user["headers"]

    session_id = str(uuid.uuid4())
    session_ref = db.collection("sessions").document(session_id)
    now = datetime.datetime.now(datetime.timezone.utc)
    session_ref.set(
        {
            "id": session_id,
            "user_id": user_id,
            "title": "Quantum Physics Exploration",
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "is_deleted": False,
            "read_status": "read",
            "last_message_content": "Assistant answer 1",
            "last_message_role": "agent",
        }
    )

    # 1. System prompt message (SHOULD BE FILTERED OUT)
    sys_msg_id = str(uuid.uuid4())
    session_ref.collection("messages").document(sys_msg_id).set(
        {
            "id": sys_msg_id,
            "session_id": session_id,
            "role": "system",
            "content": "You are Cognito. System prompt instructions: be concise.",
            "created_at": (now - datetime.timedelta(minutes=6)).isoformat(),
            "parts": [{"type": "system", "text": "System prompt instructions"}],
            "attachment_ids": [],
        }
    )

    # 2. Tool message (SHOULD BE FILTERED OUT)
    tool_msg_id = str(uuid.uuid4())
    session_ref.collection("messages").document(tool_msg_id).set(
        {
            "id": tool_msg_id,
            "session_id": session_id,
            "role": "tool",
            "content": "tool execution result trace",
            "created_at": (now - datetime.timedelta(minutes=5, seconds=30)).isoformat(),
            "parts": [{"type": "tool_result", "output": "raw tool output"}],
            "attachment_ids": [],
        }
    )

    # 3. Add valid user message with a clean part
    msg1_id = str(uuid.uuid4())
    session_ref.collection("messages").document(msg1_id).set(
        {
            "id": msg1_id,
            "session_id": session_id,
            "role": "user",
            "content": "What is quantum entanglement?",
            "created_at": (now - datetime.timedelta(minutes=5)).isoformat(),
            "parts": [{"type": "text", "text": "What is quantum entanglement?"}],
            "attachment_ids": [],
        }
    )

    # 4. Add valid agent message with text and an internal preamble part (PREAMBLE SHOULD BE SCRUBBED)
    msg2_id = str(uuid.uuid4())
    session_ref.collection("messages").document(msg2_id).set(
        {
            "id": msg2_id,
            "session_id": session_id,
            "role": "agent",
            "content": "Quantum entanglement is a phenomenon...",
            "created_at": (now - datetime.timedelta(minutes=4)).isoformat(),
            "parts": [
                {"type": "system_instruction", "text": "Internal confidential instructions"},
                {"type": "text", "text": "Quantum entanglement is a phenomenon..."},
            ],
            "attachment_ids": [],
        }
    )

    # Share the session with show_name=True
    share_res = client.post(
        f"/agent/sessions/{session_id}/share",
        headers=headers,
        json={"title": "Custom Shared Title", "show_name": True},
    )
    assert share_res.status_code == 200, share_res.text
    share_data = share_res.json()
    share_id = share_data["share_id"]
    assert share_data["title"] == "Custom Shared Title"
    assert "show_name" not in share_data
    assert share_data["author_name"] is not None
    assert share_data["message_count"] == 2  # Only user and agent messages (system and tool were scrubbed)

    # Add message 3 AFTER the share was created
    msg3_id = str(uuid.uuid4())
    session_ref.collection("messages").document(msg3_id).set(
        {
            "id": msg3_id,
            "session_id": session_id,
            "role": "user",
            "content": "Secret follow-up that should not be visible in public share",
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "parts": [{"type": "text", "text": "Secret follow-up..."}],
            "attachment_ids": [],
        }
    )

    # Fetch public chat WITHOUT any authentication
    public_res = client.get(f"/agent/shared/{share_id}")
    assert public_res.status_code == 200
    public_data = public_res.json()
    assert public_data["id"] == share_id
    assert public_data["title"] == "Custom Shared Title"
    assert "show_name" not in public_data
    assert public_data["is_owner"] is False
    assert public_data["message_count"] == 2
    assert len(public_data["messages"]) == 2

    # Owner authenticated access returns is_owner=True
    owner_res = client.get(f"/agent/shared/{share_id}", headers=headers)
    assert owner_res.status_code == 200
    assert owner_res.json()["is_owner"] is True

    # Secondary authenticated user returns is_owner=False
    sec_res = client.get(f"/agent/shared/{share_id}", headers=secondary_user["headers"])
    assert sec_res.status_code == 200
    assert sec_res.json()["is_owner"] is False

    # Verify no system parts or messages exist
    roles = [m["role"] for m in public_data["messages"]]
    assert "system" not in roles
    assert "tool" not in roles

    # Verify agent message parts were scrubbed of internal parts
    agent_msg = next(m for m in public_data["messages"] if m["role"] == "agent")
    part_types = [p["type"] for p in agent_msg["parts"]]
    assert "system_instruction" not in part_types
    assert "text" in part_types

    # Continue chat: verify exclude_from_memory is True
    continue_res = client.post(
        f"/agent/shared/{share_id}/continue",
        headers=secondary_user["headers"],
    )
    assert continue_res.status_code == 201, continue_res.text
    continue_data = continue_res.json()
    assert continue_data["exclude_from_memory"] is True
    new_session_id = continue_data["session_id"]

    # Verify new session in DB has exclude_from_memory=True
    get_new_session = client.get(
        f"/agent/sessions/{new_session_id}",
        headers=secondary_user["headers"],
    )
    assert get_new_session.status_code == 200
    assert get_new_session.json()["session"]["exclude_from_memory"] is True


def test_share_anonymous_toggle(client: TestClient, test_user: dict):
    """Tests that show_name=False sets author_name to 'Anonymous'."""
    db: firestore.Client = test_user["db"]
    user_id = test_user["user_id"]
    headers = test_user["headers"]

    session_id = str(uuid.uuid4())
    db.collection("sessions").document(session_id).set(
        {
            "id": session_id,
            "user_id": user_id,
            "title": "Anonymous Chat",
            "is_deleted": False,
        }
    )

    share_res = client.post(
        f"/agent/sessions/{session_id}/share",
        headers=headers,
        json={"show_name": False},
    )
    assert share_res.status_code == 200
    share_id = share_res.json()["share_id"]

    public_res = client.get(f"/agent/shared/{share_id}")
    assert public_res.status_code == 200
    data = public_res.json()
    assert "show_name" not in data
    assert data["author_name"] == "Anonymous"


def test_revocation_sets_revoked_at_and_returns_share_revoked(
    client: TestClient, test_user: dict, secondary_user: dict
):
    """Tests revoking a shared link: sets revoked_at, invalidates cache, and returns 404 SHARE_REVOKED."""
    db: firestore.Client = test_user["db"]
    user_id = test_user["user_id"]
    headers = test_user["headers"]

    session_id = str(uuid.uuid4())
    db.collection("sessions").document(session_id).set(
        {
            "id": session_id,
            "user_id": user_id,
            "title": "To be revoked",
            "is_deleted": False,
        }
    )

    # 1. Create share
    share_res = client.post(f"/agent/sessions/{session_id}/share", headers=headers)
    share_id = share_res.json()["share_id"]

    # 2. Verify accessible initially
    init_res = client.get(f"/agent/shared/{share_id}")
    assert init_res.status_code == 200

    # 3. Non-owner cannot revoke
    unauth_delete = client.delete(f"/agent/shared/{share_id}", headers=secondary_user["headers"])
    assert unauth_delete.status_code == 404

    # 4. Owner revokes share
    delete_res = client.delete(f"/agent/shared/{share_id}", headers=headers)
    assert delete_res.status_code == 200
    assert delete_res.json()["message"] == "Shared chat revoked successfully"

    # 5. Accessing revoked share returns 404 with SHARE_REVOKED
    revoked_res = client.get(f"/agent/shared/{share_id}")
    assert revoked_res.status_code == 404
    error_body = revoked_res.json()
    assert error_body.get("code") == "SHARE_REVOKED"
    assert "deleted" in error_body.get("detail", "").lower()

    # 6. Trying to continue a revoked share returns 404 SHARE_REVOKED
    continue_revoked = client.post(
        f"/agent/shared/{share_id}/continue",
        headers=secondary_user["headers"],
    )
    assert continue_revoked.status_code == 404
    assert continue_revoked.json().get("code") == "SHARE_REVOKED"


def test_session_level_revocation(client: TestClient, test_user: dict):
    """Tests DELETE /agent/sessions/{session_id}/share revokes all shares for that session."""
    db: firestore.Client = test_user["db"]
    user_id = test_user["user_id"]
    headers = test_user["headers"]

    session_id = str(uuid.uuid4())
    db.collection("sessions").document(session_id).set(
        {
            "id": session_id,
            "user_id": user_id,
            "title": "Session revocation test",
            "is_deleted": False,
        }
    )

    share_res = client.post(f"/agent/sessions/{session_id}/share", headers=headers)
    share_id = share_res.json()["share_id"]

    # Revoke via session endpoint
    del_res = client.delete(f"/agent/sessions/{session_id}/share", headers=headers)
    assert del_res.status_code == 200

    # Should now return 404 SHARE_REVOKED
    get_res = client.get(f"/agent/shared/{share_id}")
    assert get_res.status_code == 404
    assert get_res.json().get("code") == "SHARE_REVOKED"


def test_get_session_share_and_update(client: TestClient, test_user: dict):
    """Tests GET /agent/sessions/{session_id}/share and updating an existing shared conversation."""
    db: firestore.Client = test_user["db"]
    user_id = test_user["user_id"]
    headers = test_user["headers"]

    session_id = str(uuid.uuid4())
    session_ref = db.collection("sessions").document(session_id)
    session_ref.set(
        {
            "id": session_id,
            "user_id": user_id,
            "title": "Initial Session",
            "is_deleted": False,
        }
    )

    # 1. Before sharing: GET returns 404
    pre_res = client.get(f"/agent/sessions/{session_id}/share", headers=headers)
    assert pre_res.status_code == 404

    # Add message
    msg1_id = str(uuid.uuid4())
    session_ref.collection("messages").document(msg1_id).set(
        {
            "id": msg1_id,
            "session_id": session_id,
            "role": "user",
            "content": "Hello world",
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "parts": [{"type": "text", "text": "Hello world"}],
            "attachment_ids": [],
        }
    )

    # 2. Share session
    share_res = client.post(f"/agent/sessions/{session_id}/share", headers=headers)
    assert share_res.status_code == 200
    share_data = share_res.json()
    assert share_data["message_count"] == 1
    share_id = share_data["share_id"]

    # 3. GET /agent/sessions/{session_id}/share now returns the active share
    get_res = client.get(f"/agent/sessions/{session_id}/share", headers=headers)
    assert get_res.status_code == 200
    get_data = get_res.json()
    assert get_data["share_id"] == share_id
    assert get_data["message_count"] == 1

    # 4. Add another message and update share
    msg2_id = str(uuid.uuid4())
    session_ref.collection("messages").document(msg2_id).set(
        {
            "id": msg2_id,
            "session_id": session_id,
            "role": "agent",
            "content": "Updated response",
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "parts": [{"type": "text", "text": "Updated response"}],
            "attachment_ids": [],
        }
    )

    # Update shared chat via POST
    update_res = client.post(
        f"/agent/sessions/{session_id}/share",
        headers=headers,
        json={"title": "Updated Session Title"},
    )
    assert update_res.status_code == 200
    update_data = update_res.json()
    assert update_data["share_id"] == share_id  # Reuses same share_id
    assert update_data["message_count"] == 2
    assert update_data["title"] == "Updated Session Title"

    # GET returns updated count
    updated_get = client.get(f"/agent/sessions/{session_id}/share", headers=headers)
    assert updated_get.status_code == 200
    assert updated_get.json()["message_count"] == 2


def test_session_has_share_id_appended_and_cleared_on_revoke(client: TestClient, test_user: dict):
    """Tests that sharing a session appends share_id to the session model and revoking clears it."""
    db: firestore.Client = test_user["db"]
    user_id = test_user["user_id"]
    headers = test_user["headers"]

    session_id = str(uuid.uuid4())
    session_ref = db.collection("sessions").document(session_id)
    now = datetime.datetime.now(datetime.timezone.utc)
    session_ref.set(
        {
            "id": session_id,
            "user_id": user_id,
            "title": "Share ID Session Test",
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "is_deleted": False,
            "read_status": "read",
            "last_message_content": "Initial message",
            "last_message_role": "user",
        }
    )

    # 1. Before sharing: session has no share_id
    res_before = client.get(f"/agent/sessions/{session_id}", headers=headers)
    assert res_before.status_code == 200
    assert res_before.json()["session"]["share_id"] is None

    # 2. Share session
    share_res = client.post(f"/agent/sessions/{session_id}/share", headers=headers)
    assert share_res.status_code == 200
    share_id = share_res.json()["share_id"]

    # 3. After sharing: session has share_id appended
    res_after = client.get(f"/agent/sessions/{session_id}", headers=headers)
    assert res_after.status_code == 200
    assert res_after.json()["session"]["share_id"] == share_id

    # Also check list sessions
    list_res = client.get("/agent/sessions", headers=headers)
    assert list_res.status_code == 200
    matching = [s for s in list_res.json()["items"] if s["id"] == session_id]
    assert len(matching) == 1
    assert matching[0]["share_id"] == share_id

    # 4. Revoke share
    del_res = client.delete(f"/agent/sessions/{session_id}/share", headers=headers)
    assert del_res.status_code == 200

    # 5. After revoking: session share_id is cleared to None
    res_revoked = client.get(f"/agent/sessions/{session_id}", headers=headers)
    assert res_revoked.status_code == 200
    assert res_revoked.json()["session"]["share_id"] is None
