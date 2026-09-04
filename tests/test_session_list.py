import datetime
import uuid

import jwt
from fastapi.testclient import TestClient
from google.cloud import firestore

from app.models.chats import SESSION_LIST_PREVIEW_CHARS


def _auth_user(client: TestClient):
    email = f"list_{uuid.uuid4().hex}@example.com"
    client.post("/auth/signup", json={"email": email, "password": "password123"})
    resp = client.post("/auth/login", json={"email": email, "password": "password123"})
    token = resp.json()["access_token"]
    payload = jwt.decode(token, options={"verify_signature": False})
    return {
        "headers": {"Authorization": f"Bearer {token}"},
        "user_id": payload["sub"],
        "db": firestore.Client(project="test-project"),
    }


def _write_session(db, user_id: str, title: str, updated_at: datetime.datetime, last_message: str = "hi"):
    session_id = str(uuid.uuid4())
    db.collection("sessions").document(session_id).set(
        {
            "id": session_id,
            "user_id": user_id,
            "title": title,
            "created_at": updated_at.isoformat(),
            "updated_at": updated_at.isoformat(),
            "is_deleted": False,
            "read_status": "read",
            "last_message_content": last_message,
            "last_message_role": "agent",
        }
    )
    return session_id


def test_list_sessions_paginates_newest_first(client: TestClient):
    auth = _auth_user(client)
    db = auth["db"]
    user_id = auth["user_id"]
    now = datetime.datetime.now(datetime.timezone.utc)

    oldest = _write_session(db, user_id, "oldest", now - datetime.timedelta(minutes=3))
    middle = _write_session(db, user_id, "middle", now - datetime.timedelta(minutes=2))
    newest = _write_session(db, user_id, "newest", now - datetime.timedelta(minutes=1))
    deleted_id = _write_session(db, user_id, "trashed", now + datetime.timedelta(seconds=1))
    db.collection("sessions").document(deleted_id).update({"is_deleted": True})

    page1 = client.get("/agent/sessions?limit=2&offset=0", headers=auth["headers"])
    assert page1.status_code == 200
    body1 = page1.json()
    assert [s["title"] for s in body1["items"]] == ["newest", "middle"]
    assert body1["has_more"] is True
    assert {s["id"] for s in body1["items"]} == {newest, middle}

    page2 = client.get("/agent/sessions?limit=2&offset=2", headers=auth["headers"])
    assert page2.status_code == 200
    body2 = page2.json()
    assert [s["title"] for s in body2["items"]] == ["oldest"]
    assert body2["has_more"] is False
    assert {s["id"] for s in body2["items"]} == {oldest}
    assert all(s["id"] != deleted_id for s in body1["items"] + body2["items"])


def test_list_sessions_clips_last_message_preview(client: TestClient):
    auth = _auth_user(client)
    now = datetime.datetime.now(datetime.timezone.utc)
    long_message = "A" * 2000
    _write_session(auth["db"], auth["user_id"], "long", now, last_message=long_message)

    resp = client.get("/agent/sessions?limit=10&offset=0", headers=auth["headers"])
    assert resp.status_code == 200
    item = next(s for s in resp.json()["items"] if s["title"] == "long")
    assert item["last_message_content"] != long_message
    assert len(item["last_message_content"]) <= SESSION_LIST_PREVIEW_CHARS + 1
    assert item["last_message_content"].endswith("…")
    assert "messages" not in item
