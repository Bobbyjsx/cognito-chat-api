"""Tests for attachment upload, classification, and chat integration."""

import uuid
from unittest.mock import AsyncMock, patch

import pytest

from app.main import app
from app.models.config import AppConfigDB
from app.storage.local import LocalStorageBackend
from app.utils.mime import classify_attachment, detect_mime

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 256


# ── MIME / classification unit tests ──────────────────────────────────────────


def test_detect_mime_sniffs_magic_bytes():
    assert detect_mime("photo.bin", None, PNG_BYTES) == "image/png"
    assert detect_mime("file.bin", None, b"%PDF-1.4 x") == "application/pdf"
    assert detect_mime("file.bin", None, b"OggS\x00\x02") == "audio/ogg"


def test_detect_mime_falls_back_to_filename_and_header():
    assert detect_mime("notes.txt", None, b"plain") == "text/plain"
    assert detect_mime("noext", "application/json; charset=utf-8", b"{}") == "application/json"


def test_classify_attachment_types():
    assert classify_attachment("a.jpg", "image/jpeg").value == "image"
    assert classify_attachment("a.webm", "video/webm").value == "video"
    assert classify_attachment("a.mp3", "audio/mpeg").value == "audio"
    assert classify_attachment("a.pdf", "application/pdf").value == "pdf"
    assert classify_attachment("a.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document").value == "document"
    assert classify_attachment("a.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet").value == "spreadsheet"
    assert classify_attachment("a.json", "application/json").value == "json"
    assert classify_attachment("a.txt", "text/plain").value == "text"
    assert classify_attachment("a.py", "text/x-python").value == "document"


# ── endpoint tests (require the Firestore emulator) ──────────────────────────


@pytest.fixture
def attachment_storage(tmp_path):
    backend = LocalStorageBackend(root=str(tmp_path))
    app.dependency_overrides = {}
    from app.api.dependencies import get_storage_backend

    app.dependency_overrides[get_storage_backend] = lambda: backend
    yield backend
    app.dependency_overrides = {}


@pytest.fixture
def auth_headers(client):
    client.post("/auth/signup", json={"email": "attach@example.com", "password": "password123"})
    resp = client.post("/auth/login", json={"email": "attach@example.com", "password": "password123"})
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def test_upload_image_attachment(client, auth_headers, attachment_storage):
    resp = client.post(
        "/agent/attachments",
        headers=auth_headers,
        files={"file": ("photo.png", PNG_BYTES, "image/png")},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["type"] == "image"
    assert data["mime_type"] == "image/png"
    assert data["size"] == len(PNG_BYTES)
    assert data["storage_uri"].startswith("local://attachments/")

    fetched = client.get(f"/agent/attachments/{data['id']}", headers=auth_headers)
    assert fetched.status_code == 200
    assert fetched.json()["filename"] == "photo.png"


def test_upload_rejects_oversize(client, auth_headers, attachment_storage):
    with patch(
        "app.repositories.config.ConfigRepository.get_config",
        new=AsyncMock(return_value=AppConfigDB(attachment_max_size=10)),
    ):
        resp = client.post(
            "/agent/attachments",
            headers=auth_headers,
            files={"file": ("big.txt", b"x" * 64, "text/plain")},
        )
    assert resp.status_code == 413


def test_upload_rejects_disallowed_type(client, auth_headers, attachment_storage):
    with patch(
        "app.repositories.config.ConfigRepository.get_config",
        new=AsyncMock(return_value=AppConfigDB(attachment_allowed_types=["image"])),
    ):
        resp = client.post(
            "/agent/attachments",
            headers=auth_headers,
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )
    assert resp.status_code == 400
    assert "not allowed" in resp.json()["detail"]


def test_upload_requires_auth(client, attachment_storage):
    resp = client.post("/agent/attachments", files={"file": ("a.png", PNG_BYTES, "image/png")})
    assert resp.status_code == 401


def test_chat_with_attachment_persists_and_prepares_text(client, auth_headers, attachment_storage, mock_agent):
    upload = client.post(
        "/agent/attachments",
        headers=auth_headers,
        files={"file": ("notes.txt", b"the answer is 42", "text/plain")},
    )
    attachment_id = upload.json()["id"]

    chat = client.post(
        "/agent/chat",
        headers=auth_headers,
        json={"message": "What does this say?", "attachments": [attachment_id]},
    )
    assert chat.status_code == 200
    session_id = chat.json()["session_id"]

    detail = client.get(f"/agent/sessions/{session_id}", headers=auth_headers).json()
    user_message = detail["messages"][0]
    assert user_message["role"] == "user"
    assert attachment_id in user_message["attachment_ids"]

    # Attachment is now bound to the session
    listing = client.get(f"/agent/attachments?session_id={session_id}", headers=auth_headers).json()
    assert listing[0]["id"] == attachment_id


def test_chat_with_unknown_attachment_rejected(client, auth_headers, attachment_storage, mock_agent):
    resp = client.post(
        "/agent/chat",
        headers=auth_headers,
        json={"message": "hi", "attachments": [str(uuid.uuid4())]},
    )
    assert resp.status_code == 400
    assert "not found" in resp.json()["detail"]


def test_delete_attachment(client, auth_headers, attachment_storage):
    upload = client.post(
        "/agent/attachments",
        headers=auth_headers,
        files={"file": ("del.txt", b"bye", "text/plain")},
    )
    attachment_id = upload.json()["id"]

    deleted = client.delete(f"/agent/attachments/{attachment_id}", headers=auth_headers)
    assert deleted.status_code == 200

    fetched = client.get(f"/agent/attachments/{attachment_id}", headers=auth_headers)
    assert fetched.status_code == 404
