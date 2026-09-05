"""Attachment upload and management endpoints.

Uploads store bytes in the configured object storage backend; only metadata
lands in Firestore. Chat requests reference attachments by id (see
``ChatRequest.attachments``).
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, Response, UploadFile
from google.cloud.firestore_v1.async_client import AsyncClient

from app.api.dependencies import (
    get_attachment_url_service,
    get_current_user,
    get_provider,
    get_storage_backend,
)
from app.database import get_db
from app.models.attachments import AttachmentSchema, PresignedUploadRequest, PresignedUploadResponse
from app.models.pagination import PaginatedResponse
from app.models.users import UserDB
from app.providers.base import BaseProvider
from app.repositories.attachments import AttachmentRepository
from app.repositories.config import ConfigRepository
from app.services.attachment_url import AttachmentUrlService
from app.services.attachments import AttachmentService
from app.storage.base import StorageBackend

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["attachments"])


def get_attachment_service(
    db: AsyncClient = Depends(get_db),
    provider: BaseProvider = Depends(get_provider),
    storage: StorageBackend = Depends(get_storage_backend),
) -> AttachmentService:
    return AttachmentService(AttachmentRepository(db), storage, provider)


@router.post("/attachments/upload-url", response_model=PresignedUploadResponse, status_code=200)
async def request_upload_url(
    body: PresignedUploadRequest,
    current_user: UserDB = Depends(get_current_user),
    db: AsyncClient = Depends(get_db),
    service: AttachmentService = Depends(get_attachment_service),
):
    """Generate a presigned upload URL for direct client-to-storage upload."""
    if body.session_id:
        from app.repositories.chats import ChatRepository

        if not await ChatRepository(db).session_exists(body.session_id, current_user.id):
            raise HTTPException(status_code=404, detail="Session not found.")

    config = await ConfigRepository(db).get_config()
    try:
        return await service.create_upload_ticket(
            user=current_user,
            filename=body.filename,
            content_type=body.mime_type,
            size=body.size,
            config=config,
            session_id=body.session_id,
            is_temporary=body.is_temporary,
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unexpected error generating upload ticket")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.") from None


@router.put("/attachments/direct-upload", status_code=200)
async def direct_upload(
    token: str,
    request: Request,
    storage: StorageBackend = Depends(get_storage_backend),
):
    """Direct upload endpoint for local storage in dev/test."""
    from app.core.security import verify_storage_token

    payload = verify_storage_token(token)
    if not payload or payload.get("action") != "upload":
        raise HTTPException(status_code=403, detail="Invalid or expired storage token")

    key = payload["key"]
    content_type = payload.get("content_type", "application/octet-stream")
    data = await request.body()
    await storage.upload_bytes(key, data, content_type)
    return {"message": "Direct upload successful", "size": len(data)}


@router.get("/attachments/direct-content")
async def direct_content(
    token: str,
    storage: StorageBackend = Depends(get_storage_backend),
):
    """Direct download endpoint for local storage in dev/test."""
    from app.core.security import verify_storage_token

    payload = verify_storage_token(token)
    if not payload or payload.get("action") != "download":
        raise HTTPException(status_code=403, detail="Invalid or expired storage token")

    uri = payload.get("uri") or f"local://{payload['key']}"
    try:
        content = await storage.read_bytes(uri)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Attachment file not found")
    headers = {}
    if payload.get("disposition") == "attachment":
        filename = payload.get("filename", "download")
        headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    return Response(content=content, headers=headers)


@router.post("/attachments", response_model=AttachmentSchema, status_code=201)
async def upload_attachment(
    file: UploadFile = File(..., description="File to upload"),
    session_id: uuid.UUID | None = Form(default=None, description="Optional session this attachment belongs to"),
    is_temporary: bool = Form(default=True, description="Whether this is a temporary upload"),
    current_user: UserDB = Depends(get_current_user),
    db: AsyncClient = Depends(get_db),
    service: AttachmentService = Depends(get_attachment_service),
):
    """Upload a file attachment.

    Supported types: images (JPEG/PNG/WebP/GIF), PDF, documents
    (TXT/Markdown/CSV/DOCX/source code), spreadsheets (XLSX), audio
    (MP3/WAV/M4A/OGG/WebM) and video (MP4/MOV/WebM).

    The file bytes are stored in object storage; Firestore keeps only metadata.
    Pass the returned ``id`` in ``attachments`` on ``POST /agent/chat`` or
    ``POST /agent/chat/stream`` to attach it to a message.
    """
    if session_id:
        from app.repositories.chats import ChatRepository

        if not await ChatRepository(db).session_exists(session_id, current_user.id):
            raise HTTPException(status_code=404, detail="Session not found.")

    data = await file.read()
    config = await ConfigRepository(db).get_config()

    if len(data) > config.attachment_max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File is too large. Maximum size is {config.attachment_max_size} bytes.",
        )

    try:
        return await service.ingest(
            user=current_user,
            session_id=session_id,
            filename=file.filename or "unnamed",
            content_type=file.content_type,
            data=data,
            config=config,
            is_temporary=is_temporary,
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unexpected error uploading attachment")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.") from None


@router.get("/attachments", response_model=PaginatedResponse[AttachmentSchema])
async def list_attachments(
    session_id: uuid.UUID | None = None,
    type: str | None = None,
    query: str | None = None,
    limit: int = 15,
    offset: int = 0,
    current_user: UserDB = Depends(get_current_user),
    db: AsyncClient = Depends(get_db),
    url_service: AttachmentUrlService = Depends(get_attachment_url_service),
):
    from app.core.cache_keys import CacheKeys
    from app.core.redis import redis_cache

    cache_key = CacheKeys.user_attachments(current_user.id, limit, offset, session_id, type, query)
    cached_data = await redis_cache.get(cache_key)
    if cached_data:
        return cached_data

    repo = AttachmentRepository(db)
    metadata, has_more, total = await repo.list_by_user(
        user_id=current_user.id, session_id=session_id, type=type, query_string=query, limit=limit, offset=offset
    )
    items = await url_service.enrich_attachments(metadata)
    response = PaginatedResponse(items=items, total=total, limit=limit, offset=offset, has_more=has_more)

    await redis_cache.set(cache_key, response.model_dump(mode="json"), expire=300)
    return response


@router.get("/attachments/{attachment_id}", response_model=AttachmentSchema)
async def get_attachment(
    attachment_id: uuid.UUID,
    current_user: UserDB = Depends(get_current_user),
    db: AsyncClient = Depends(get_db),
    url_service: AttachmentUrlService = Depends(get_attachment_url_service),
):
    repo = AttachmentRepository(db)
    metadata = await repo.get(attachment_id, current_user.id)
    if metadata is None:
        raise HTTPException(status_code=404, detail="Attachment not found")
    return await url_service.enrich_attachment(metadata)


@router.get("/attachments/{attachment_id}/content")
async def get_attachment_content(
    attachment_id: uuid.UUID,
    current_user: UserDB = Depends(get_current_user),
    db: AsyncClient = Depends(get_db),
    service: AttachmentService = Depends(get_attachment_service),
):
    repo = AttachmentRepository(db)
    metadata = await repo.get(attachment_id, current_user.id)
    if metadata is None:
        raise HTTPException(status_code=404, detail="Attachment not found")

    content = await service.read_bytes(metadata)
    headers = {"Content-Disposition": f'attachment; filename="{metadata.filename}"'}
    return Response(content=content, media_type=metadata.mime_type, headers=headers)


@router.delete("/attachments/{attachment_id}", status_code=200)
async def delete_attachment(
    attachment_id: uuid.UUID,
    current_user: UserDB = Depends(get_current_user),
    service: AttachmentService = Depends(get_attachment_service),
):
    deleted = await service.delete(current_user.id, attachment_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Attachment not found")
    return {"message": "Attachment deleted successfully"}
