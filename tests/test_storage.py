"""Tests for the local storage backend."""

import pytest

from app.storage.local import LocalStorageBackend


@pytest.mark.asyncio
async def test_upload_read_delete_roundtrip(tmp_path):
    backend = LocalStorageBackend(root=str(tmp_path))

    uri = await backend.upload_bytes("attachments/u1/img.png", b"image-bytes", "image/png")
    assert uri == "local://attachments/u1/img.png"

    assert await backend.read_bytes(uri) == b"image-bytes"

    await backend.delete(uri)
    with pytest.raises(FileNotFoundError):
        await backend.read_bytes(uri)


@pytest.mark.asyncio
async def test_path_traversal_is_rejected(tmp_path):
    backend = LocalStorageBackend(root=str(tmp_path))
    with pytest.raises(ValueError):
        await backend.read_bytes("local://../../etc/passwd")


@pytest.mark.asyncio
async def test_delete_missing_is_noop(tmp_path):
    backend = LocalStorageBackend(root=str(tmp_path))
    await backend.delete("local://does/not/exist")


@pytest.mark.asyncio
async def test_presigned_urls_local_backend(tmp_path):
    backend = LocalStorageBackend(root=str(tmp_path))

    upload_url, headers = await backend.generate_upload_url("attachments/u1/doc.pdf", "application/pdf")
    assert "/agent/attachments/direct-upload?token=" in upload_url
    assert headers == {"Content-Type": "application/pdf"}

    download_url = await backend.generate_download_url("local://attachments/u1/doc.pdf")
    assert "/agent/attachments/direct-content?token=" in download_url


@pytest.mark.asyncio
async def test_attachment_service_upload_ticket_and_url(tmp_path):
    from unittest.mock import AsyncMock, MagicMock

    from app.models.config import AppConfigDB
    from app.models.users import UserDB
    from app.services.attachments import AttachmentService

    backend = LocalStorageBackend(root=str(tmp_path))
    repo = MagicMock()
    repo.create = AsyncMock()
    service = AttachmentService(repo=repo, storage=backend, provider=MagicMock())

    user = UserDB(id="00000000-0000-0000-0000-000000000001", email="test@example.com", hashed_password="pw")
    config = AppConfigDB()

    ticket = await service.create_upload_ticket(
        user=user,
        filename="photo.png",
        content_type="image/png",
        size=100,
        config=config,
    )
    assert ticket.upload_url.startswith("/agent/attachments/direct-upload?token=")
    assert ticket.attachment.url is not None
    assert ticket.attachment.url.startswith("/agent/attachments/direct-content?token=")
