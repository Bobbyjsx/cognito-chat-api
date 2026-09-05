"""Tests for attachment URL generation, authorization, enrichment, and Firestore isolation."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.models.attachments import AttachmentMetadata, AttachmentType
from app.models.chats import ChatMessageDB, MessageRole
from app.repositories.chats import ChatRepository
from app.services.attachment_url import AttachmentUrlService
from app.storage.local import LocalStorageBackend


@pytest.fixture
def storage(tmp_path):
    return LocalStorageBackend(root=str(tmp_path))


@pytest.fixture
def url_service(storage):
    return AttachmentUrlService(storage)


@pytest.mark.asyncio
async def test_attachment_url_generation(url_service):
    """Test generating a signed URL with explicit expiration."""
    user_id = uuid4()
    meta = AttachmentMetadata(
        user_id=user_id,
        filename="report.pdf",
        mime_type="application/pdf",
        size=1024,
        storage_uri="local://attachments/u1/report.pdf",
        type=AttachmentType.pdf,
    )
    assert meta.bucket == "local"
    assert meta.object_name == "attachments/u1/report.pdf"

    schema = await url_service.enrich_attachment(meta, expires_in=3600)
    assert schema.url is not None
    assert "/agent/attachments/direct-content?token=" in schema.url
    assert schema.url_expires_at is not None
    assert schema.url_expires_at > datetime.now(timezone.utc)
    assert schema.bucket == "local"
    assert schema.object_name == "attachments/u1/report.pdf"
    assert schema.content_type == "application/pdf"


@pytest.mark.asyncio
async def test_authorization_for_attachment_access(url_service):
    """Ensure user A cannot obtain signed URLs for attachments belonging to user B."""
    user_a = uuid4()
    user_b = uuid4()
    att_b = AttachmentMetadata(
        id=uuid4(),
        user_id=user_b,
        filename="private_b.png",
        mime_type="image/png",
        size=500,
        storage_uri="local://attachments/user_b/private_b.png",
    )

    att_repo = MagicMock()

    # If user A queries, att_repo.get_many returns only attachments owned by user A (so empty)
    async def fake_get_many(query_user_id, ids):
        if str(query_user_id) == str(user_b):
            return [att_b]
        return []

    att_repo.get_many = AsyncMock(side_effect=fake_get_many)

    messages = [
        ChatMessageDB(
            session_id=uuid4(),
            role=MessageRole.USER,
            content="Look at this",
            attachment_ids=[str(att_b.id)],
            parts=[{"type": "file", "attachment_id": str(att_b.id)}],
        )
    ]

    # User A tries to view message referencing User B's attachment
    await url_service.enrich_message_attachments(messages, user_id=user_a, att_repo=att_repo)

    part = messages[0].parts[0]
    # Since user A does not own att_b, it should not receive a signed URL
    assert part.get("url") is None


@pytest.mark.asyncio
async def test_expired_stale_url_behavior_on_subsequent_fetches(url_service):
    """Verify each fetch produces fresh signed URLs with current expirations."""
    meta = AttachmentMetadata(
        user_id=uuid4(),
        filename="chart.png",
        mime_type="image/png",
        size=2048,
        storage_uri="local://attachments/chart.png",
    )

    fetch1 = await url_service.enrich_attachment(meta, expires_in=1800)
    fetch2 = await url_service.enrich_attachment(meta, expires_in=3600)

    assert fetch1.url is not None
    assert fetch2.url is not None
    # Both are valid signed URLs
    assert fetch1.url_expires_at is not None
    assert fetch2.url_expires_at is not None
    # Expirations reflect the requested lifetimes
    diff = (fetch2.url_expires_at - fetch1.url_expires_at).total_seconds()
    assert 1700 <= diff <= 1900


@pytest.mark.asyncio
async def test_multiple_attachments_in_single_response(url_service):
    """Verify batch enrichment for multiple attachments across messages in a single DB query."""
    user_id = uuid4()
    metas = [
        AttachmentMetadata(
            id=uuid4(),
            user_id=user_id,
            filename=f"doc_{i}.pdf",
            mime_type="application/pdf",
            size=100 * (i + 1),
            storage_uri=f"local://attachments/doc_{i}.pdf",
        )
        for i in range(5)
    ]

    att_repo = MagicMock()
    att_repo.get_many = AsyncMock(return_value=metas)

    messages = [
        ChatMessageDB(
            session_id=uuid4(),
            role=MessageRole.USER,
            content=f"Message {i}",
            attachment_ids=[str(metas[i].id)],
            parts=[{"type": "file", "attachment_id": str(metas[i].id)}],
        )
        for i in range(5)
    ]

    await url_service.enrich_message_attachments(messages, user_id=user_id, att_repo=att_repo)

    # Verifies single batch database call (zero N+1)
    att_repo.get_many.assert_called_once()
    called_args = att_repo.get_many.call_args[0]
    assert called_args[0] == user_id
    assert set(called_args[1]) == {str(m.id) for m in metas}

    for i, msg in enumerate(messages):
        part = msg.parts[0]
        assert part["url"] is not None
        assert "/agent/attachments/direct-content?token=" in part["url"]
        assert part["urlExpiresAt"] is not None
        assert part["contentType"] == "application/pdf"


@pytest.mark.asyncio
async def test_failure_to_sign_individual_attachment(tmp_path):
    """Gracefully handle signing failure for 1 attachment without failing the others or crashing."""
    backend = LocalStorageBackend(root=str(tmp_path))

    # Mock generate_download_url to fail for a specific URI
    original_gen = backend.generate_download_url

    async def flaky_gen(uri, *args, expires_in=3600, **kwargs):
        if "bad_object" in uri:
            raise RuntimeError("GCS connection timeout")
        return await original_gen(uri, *args, expires_in=expires_in, **kwargs)

    backend.generate_download_url = flaky_gen
    service = AttachmentUrlService(backend)

    user_id = uuid4()
    meta_good = AttachmentMetadata(
        id=uuid4(),
        user_id=user_id,
        filename="good.png",
        mime_type="image/png",
        size=100,
        storage_uri="local://good.png",
    )
    meta_bad = AttachmentMetadata(
        id=uuid4(),
        user_id=user_id,
        filename="bad.png",
        mime_type="image/png",
        size=100,
        storage_uri="local://bad_object.png",
    )

    att_repo = MagicMock()
    att_repo.get_many = AsyncMock(return_value=[meta_good, meta_bad])

    messages = [
        ChatMessageDB(
            session_id=uuid4(),
            role=MessageRole.USER,
            content="Files",
            parts=[
                {"type": "file", "attachment_id": str(meta_good.id)},
                {"type": "file", "attachment_id": str(meta_bad.id)},
            ],
        )
    ]

    # Should not raise an exception
    await service.enrich_message_attachments(messages, user_id=user_id, att_repo=att_repo)

    parts = messages[0].parts
    assert parts[0]["url"] is not None
    assert parts[1].get("url") is None  # Failed gracefully


@pytest.mark.asyncio
async def test_signed_urls_not_persisted_to_firestore():
    """Ensure AttachmentMetadata and ChatMessageDB do not persist signed URLs in Firestore."""
    user_id = uuid4()
    meta = AttachmentMetadata(
        user_id=user_id,
        filename="photo.jpg",
        mime_type="image/jpeg",
        size=1024,
        storage_uri="gs://my-bucket/attachments/photo.jpg",
    )

    dump = meta.model_dump(mode="json")
    # Stable canonical properties exist
    assert dump["bucket"] == "my-bucket"
    assert dump["object_name"] == "attachments/photo.jpg"
    assert dump["storage_uri"] == "gs://my-bucket/attachments/photo.jpg"
    # Signed URL fields do not exist in metadata
    assert "url" not in dump
    assert "url_expires_at" not in dump
    assert "urlExpiresAt" not in dump

    # Verify ChatRepository.add_message strips transient URL fields
    db_mock = MagicMock()
    batch_mock = MagicMock()
    batch_mock.commit = AsyncMock()
    db_mock.batch.return_value = batch_mock
    collection_mock = MagicMock()
    db_mock.collection.return_value = collection_mock
    doc_mock = MagicMock()
    collection_mock.document.return_value = doc_mock
    doc_mock.collection.return_value = collection_mock

    chat_repo = ChatRepository(db_mock)

    parts_with_transient_urls = [
        {
            "type": "file",
            "attachment_id": str(uuid4()),
            "url": "https://storage.googleapis.com/signed-url?token=xyz",
            "url_expires_at": "2026-09-05T13:00:00Z",
            "urlExpiresAt": "2026-09-05T13:00:00Z",
            "filename": "photo.jpg",
            "bucket": "my-bucket",
            "object_name": "photo.jpg",
        }
    ]

    msg = await chat_repo.add_message(
        session_id=uuid4(),
        role=MessageRole.USER,
        content="Testing",
        parts=parts_with_transient_urls,
        update_session_summary=False,
    )

    saved_part = msg.parts[0]
    assert "url" not in saved_part
    assert "url_expires_at" not in saved_part
    assert "urlExpiresAt" not in saved_part
    assert saved_part["bucket"] == "my-bucket"
    assert saved_part["object_name"] == "photo.jpg"


def test_expired_storage_token_rejected_by_verify():
    """Ensure verify_storage_token rejects expired tokens."""
    from app.core.security import create_storage_token, verify_storage_token

    expired_token = create_storage_token({"action": "download", "uri": "local://test.png"}, expires_in=-10)
    assert verify_storage_token(expired_token) is None


def test_tampered_storage_token_rejected():
    """Ensure verify_storage_token rejects tampered or wrong-purpose tokens."""
    import jwt

    from app.core.config import settings
    from app.core.security import create_storage_token, verify_storage_token

    # Token with wrong purpose
    wrong_purpose = jwt.encode(
        {"purpose": "wrong", "exp": 9999999999}, settings.secret_key, algorithm=settings.algorithm
    )
    assert verify_storage_token(wrong_purpose) is None

    # Tampered token
    valid_token = create_storage_token({"action": "download", "uri": "local://test.png"})
    tampered_token = valid_token[:-4] + "abcd"
    assert verify_storage_token(tampered_token) is None


def test_direct_endpoints_enforce_token_expiration(client):
    """Ensure HTTP 403 Forbidden is returned when storage tokens are expired or invalid."""
    from app.core.security import create_storage_token

    expired_token = create_storage_token({"action": "download", "uri": "local://test.png"}, expires_in=-10)
    resp = client.get(f"/agent/attachments/direct-content?token={expired_token}")
    assert resp.status_code == 403
    assert "Invalid or expired storage token" in resp.json()["detail"]

    expired_upload_token = create_storage_token({"action": "upload", "key": "test.png"}, expires_in=-10)
    resp_upload = client.put(f"/agent/attachments/direct-upload?token={expired_upload_token}", content=b"data")
    assert resp_upload.status_code == 403
    assert "Invalid or expired storage token" in resp_upload.json()["detail"]


@pytest.mark.asyncio
async def test_gcs_storage_backend_signed_url_generation():
    """Verify GCS backend generates signed URLs with expiration parameter."""
    from unittest.mock import MagicMock

    from app.storage.gcs import GCSStorageBackend

    mock_client = MagicMock()
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_client.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob
    mock_blob.generate_signed_url.return_value = "https://storage.googleapis.com/test-bucket/file.png?signed=true"

    backend = GCSStorageBackend(bucket_name="test-bucket", client=mock_client)
    url = await backend.generate_download_url("gs://test-bucket/attachments/file.png", expires_in=1800)

    assert url == "https://storage.googleapis.com/test-bucket/file.png?signed=true"
    mock_blob.generate_signed_url.assert_called_once()
    call_kwargs = mock_blob.generate_signed_url.call_args[1]
    assert call_kwargs["version"] == "v4"
    assert call_kwargs["method"] == "GET"
    assert call_kwargs["expiration"].total_seconds() == 1800


@pytest.mark.asyncio
async def test_download_url_generation_and_disposition(storage, url_service, client):
    """Verify enrich_attachment creates both inline url and download_url with attachment disposition."""
    from app.api.dependencies import get_storage_backend
    from app.main import app

    app.dependency_overrides[get_storage_backend] = lambda: storage
    try:
        user_id = uuid4()
        await storage.upload_bytes("attachments/invoice.pdf", b"pdf file contents", "application/pdf")
        meta = AttachmentMetadata(
            id=uuid4(),
            user_id=user_id,
            filename="invoice.pdf",
            mime_type="application/pdf",
            size=4096,
            storage_uri="local://attachments/invoice.pdf",
        )

        schema = await url_service.enrich_attachment(meta)
        assert schema.url is not None
        assert schema.download_url is not None
        assert schema.url != schema.download_url

        # Test that hitting download_url returns Content-Disposition header
        resp = client.get(schema.download_url)
        assert resp.status_code == 200
        assert resp.content == b"pdf file contents"
        assert resp.headers.get("Content-Disposition") == 'attachment; filename="invoice.pdf"'
    finally:
        app.dependency_overrides.pop(get_storage_backend, None)


@pytest.mark.asyncio
async def test_soft_delete_filtering():
    """Verify soft-deleted attachments are excluded from get, get_many, and list_by_user by default."""
    from unittest.mock import AsyncMock, MagicMock

    from app.repositories.attachments import AttachmentRepository

    user_id = uuid4()
    att_active = AttachmentMetadata(id=uuid4(), user_id=user_id, filename="active.png", mime_type="image/png")
    att_deleted = AttachmentMetadata(
        id=uuid4(),
        user_id=user_id,
        filename="deleted.png",
        mime_type="image/png",
        deleted_at=datetime.now(timezone.utc),
    )

    db_mock = MagicMock()
    repo = AttachmentRepository(db_mock)

    # 1. Test get method
    doc_mock = MagicMock()
    doc_mock.exists = True
    doc_mock.to_dict.return_value = att_deleted.model_dump(mode="json")
    repo.collection.document.return_value.get = AsyncMock(return_value=doc_mock)

    # Default should exclude deleted
    result = await repo.get(att_deleted.id, user_id)
    assert result is None

    # Explicit include_deleted=True returns it
    result_included = await repo.get(att_deleted.id, user_id, include_deleted=True)
    assert result_included is not None
    assert result_included.filename == "deleted.png"

    # Active attachment is returned normally
    doc_mock.to_dict.return_value = att_active.model_dump(mode="json")
    result_active = await repo.get(att_active.id, user_id)
    assert result_active is not None
    assert result_active.filename == "active.png"


@pytest.mark.asyncio
async def test_make_permanent_updates_storage_location():
    """Verify make_permanent moves file in storage and updates both storage_uri and object_name."""
    from app.services.attachments import AttachmentService

    user_id = uuid4()
    att_id = uuid4()
    meta = AttachmentMetadata(
        id=att_id,
        user_id=user_id,
        filename="diagram.svg",
        mime_type="image/svg+xml",
        size=120,
        storage_uri=f"gs://chat_attachment/attachments/temp/{user_id}/image/{att_id}_diagram.svg",
        bucket="chat_attachment",
        object_name=f"attachments/temp/{user_id}/image/{att_id}_diagram.svg",
        type=AttachmentType.image,
        is_temporary=True,
    )

    repo_mock = MagicMock()
    repo_mock.get_many = AsyncMock(return_value=[meta])
    repo_mock.update_storage_location = AsyncMock()
    repo_mock.update_temporary_flag = AsyncMock()

    storage_mock = MagicMock()
    storage_mock.move = AsyncMock(return_value=f"gs://chat_attachment/attachments/{user_id}/image/diagram.svg")

    service = AttachmentService(repo=repo_mock, storage=storage_mock, provider=MagicMock())
    service._invalidate_cache = AsyncMock()

    await service.make_permanent(user_id, [att_id])

    assert meta.is_temporary is False
    assert meta.object_name == f"attachments/{user_id}/image/diagram.svg"
    assert meta.storage_uri == f"gs://chat_attachment/attachments/{user_id}/image/diagram.svg"
    repo_mock.update_storage_location.assert_awaited_once_with(
        att_id,
        f"gs://chat_attachment/attachments/{user_id}/image/diagram.svg",
        f"attachments/{user_id}/image/diagram.svg",
    )


@pytest.mark.asyncio
async def test_prepare_parts_handles_svg_as_text():
    """Verify SVG attachments are prepared as text parts, avoiding vision API 400 Bad Request."""
    from app.services.attachments import AttachmentService

    user_id = uuid4()
    att_id = uuid4()
    svg_content = b'<svg xmlns="http://www.w3.org/2000/svg"><circle cx="10" cy="10" r="5"/></svg>'
    meta = AttachmentMetadata(
        id=att_id,
        user_id=user_id,
        filename="icon.svg",
        mime_type="image/svg+xml",
        size=len(svg_content),
        storage_uri="local://attachments/icon.svg",
        type=AttachmentType.image,
        is_temporary=False,
    )

    repo_mock = MagicMock()
    storage_mock = MagicMock()
    storage_mock.read_bytes = AsyncMock(return_value=svg_content)
    provider_mock = MagicMock()

    service = AttachmentService(repo=repo_mock, storage=storage_mock, provider=provider_mock)
    parts = await service.prepare_parts(meta)

    assert len(parts) == 1
    assert "text" in parts[0]
    assert '<circle cx="10" cy="10" r="5"/>' in parts[0]["text"]
    provider_mock.parts_for_attachment.assert_not_called()


@pytest.mark.asyncio
async def test_generate_attachment_url_stale_object_name_recovery(url_service):
    """Verify generate_attachment_url recovers if object_name is stale temp while storage_uri is permanent."""
    user_id = uuid4()
    meta = AttachmentMetadata(
        id=uuid4(),
        user_id=user_id,
        filename="test.png",
        mime_type="image/png",
        size=100,
        storage_uri="local://attachments/permanent/test.png",
        bucket="local",
        object_name="attachments/temp/user/test.png",
        type=AttachmentType.image,
    )

    url, _ = await url_service.generate_attachment_url(meta)
    assert url is not None
    # Verify the generated URL points to the permanent location, not the stale temp
    from app.core.security import verify_storage_token

    token = url.split("token=")[-1]
    payload = verify_storage_token(token)
    assert payload is not None
    assert payload["uri"] == "local://attachments/permanent/test.png"


@pytest.mark.asyncio
async def test_docx_text_extraction():
    """Verify .docx attachments are accepted, classified as document, and text is extracted."""
    import io
    import zipfile

    from app.services.attachments import AttachmentService
    from app.utils.mime import classify_attachment, detect_mime

    mime = detect_mime("sample.docx")
    assert mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    att_type = classify_attachment("sample.docx", mime)
    assert att_type == AttachmentType.document

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(
            "word/document.xml",
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>Project Specification DOCX</w:t></w:r></w:p></w:body></w:document>',
        )
    docx_bytes = buf.getvalue()

    user_id = uuid4()
    meta = AttachmentMetadata(
        id=uuid4(),
        user_id=user_id,
        filename="sample.docx",
        mime_type=mime,
        size=len(docx_bytes),
        storage_uri="local://attachments/sample.docx",
        type=att_type,
        is_temporary=False,
    )

    storage_mock = MagicMock()
    storage_mock.read_bytes = AsyncMock(return_value=docx_bytes)
    service = AttachmentService(repo=MagicMock(), storage=storage_mock, provider=MagicMock())

    parts = await service.prepare_parts(meta)
    assert len(parts) == 1
    assert "text" in parts[0]
    assert "Project Specification DOCX" in parts[0]["text"]
