"""Attachment URL and download signing service.

Handles temporary GCS signed URL generation, enrichment, and batch signing
without persisting temporary URLs in Firestore.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from app.models.attachments import AttachmentMetadata, AttachmentSchema
from app.storage.base import StorageBackend

logger = logging.getLogger(__name__)

# Default signed URL lifetime: 60 minutes (aligned with 30-60 min requirement)
DEFAULT_SIGNED_URL_EXPIRY_SECONDS = 3600


class AttachmentUrlService:
    """Service for enriching attachments with temporary signed download URLs."""

    def __init__(self, storage: StorageBackend):
        self.storage = storage

    async def generate_attachment_url(
        self,
        attachment: AttachmentMetadata | AttachmentSchema | dict[str, Any] | str,
        expires_in: int = DEFAULT_SIGNED_URL_EXPIRY_SECONDS,
        filename: str | None = None,
        disposition: str = "inline",
    ) -> tuple[str | None, datetime | None]:
        """Generate a fresh signed download URL for an attachment target.

        Uses stored bucket + object_name (or storage_uri).
        Handles individual signing errors gracefully so caller flows are not interrupted.
        """
        target: str | None = None
        if isinstance(attachment, (AttachmentMetadata, AttachmentSchema)):
            filename = filename or attachment.filename
            if (
                attachment.storage_uri
                and "temp/" not in attachment.storage_uri
                and attachment.object_name
                and "temp/" in attachment.object_name
            ):
                target = attachment.storage_uri
            elif attachment.bucket and attachment.object_name:
                if attachment.bucket == "local":
                    target = f"local://{attachment.object_name}"
                else:
                    target = f"gs://{attachment.bucket}/{attachment.object_name}"
            elif attachment.storage_uri:
                target = attachment.storage_uri
        elif isinstance(attachment, dict):
            filename = filename or attachment.get("filename")
            bucket = attachment.get("bucket")
            object_name = attachment.get("object_name") or attachment.get("objectName")
            storage_uri = attachment.get("storage_uri") or attachment.get("storageUri")
            if storage_uri and "temp/" not in storage_uri and object_name and "temp/" in object_name:
                target = storage_uri
            elif bucket and object_name:
                target = f"local://{object_name}" if bucket == "local" else f"gs://{bucket}/{object_name}"
            else:
                target = storage_uri
        elif isinstance(attachment, str):
            target = attachment

        if not target:
            return None, None

        expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
        try:
            url = await self.storage.generate_download_url(
                target,
                expires_in=expires_in,
                filename=filename,
                disposition=disposition,
            )
            return url, expires_at
        except Exception as e:
            att_id = getattr(attachment, "id", None) or (
                attachment.get("id") if isinstance(attachment, dict) else target
            )
            logger.warning("Failed to generate signed download URL for attachment %s: %s", att_id, e)
            return None, None

    async def enrich_attachment(
        self,
        metadata: AttachmentMetadata,
        expires_in: int = DEFAULT_SIGNED_URL_EXPIRY_SECONDS,
    ) -> AttachmentSchema:
        """Convert AttachmentMetadata to wire schema and attach fresh signed URLs."""
        schema = AttachmentSchema.model_validate(metadata, from_attributes=True)
        url, expires_at = await self.generate_attachment_url(metadata, expires_in=expires_in, disposition="inline")
        download_url, _ = await self.generate_attachment_url(metadata, expires_in=expires_in, disposition="attachment")
        schema.url = url
        schema.download_url = download_url
        schema.url_expires_at = expires_at
        return schema

    async def enrich_attachments(
        self,
        metadatas: Sequence[AttachmentMetadata],
        expires_in: int = DEFAULT_SIGNED_URL_EXPIRY_SECONDS,
    ) -> list[AttachmentSchema]:
        """Enrich multiple attachments concurrently with fresh signed URLs."""
        if not metadatas:
            return []

        async def _enrich(m: AttachmentMetadata) -> AttachmentSchema:
            return await self.enrich_attachment(m, expires_in=expires_in)

        return await asyncio.gather(*[_enrich(m) for m in metadatas])

    async def enrich_message_attachments(
        self,
        messages: Sequence[Any],
        user_id: UUID | str,
        att_repo: Any,
        expires_in: int = DEFAULT_SIGNED_URL_EXPIRY_SECONDS,
    ) -> None:
        """Enrich a conversation/session's message list with fresh signed URLs.

        Retrieves all attachments in a single batch Firestore query (no N+1),
        enforces user authorization, and signs URLs concurrently.
        """
        if not messages:
            return

        all_att_ids: set[str] = set()
        for msg in messages:
            for aid in getattr(msg, "attachment_ids", []) or []:
                all_att_ids.add(str(aid))
            for part in getattr(msg, "parts", []) or []:
                if isinstance(part, dict) and part.get("type") == "file" and part.get("attachment_id"):
                    all_att_ids.add(str(part["attachment_id"]))

        if not all_att_ids:
            return

        # 1. Single database retrieval path with ownership filter
        att_metas = await att_repo.get_many(user_id, list(all_att_ids))
        att_map: dict[str, AttachmentMetadata] = {str(m.id): m for m in att_metas}

        # 2. Concurrently sign all URLs (both inline display and attachment download)
        url_map: dict[str, tuple[str | None, str | None, datetime | None]] = {}

        async def _sign(aid: str, meta: AttachmentMetadata):
            url, exp = await self.generate_attachment_url(meta, expires_in=expires_in, disposition="inline")
            dl_url, _ = await self.generate_attachment_url(
                meta, expires_in=expires_in, filename=meta.filename, disposition="attachment"
            )
            url_map[aid] = (url, dl_url, exp)

        await asyncio.gather(*[_sign(aid, meta) for aid, meta in att_map.items()])

        # 3. Enrich message parts in place
        for msg in messages:
            msg_parts = list(getattr(msg, "parts", []) or [])
            has_file_parts = False
            for part in msg_parts:
                if isinstance(part, dict) and part.get("type") == "file":
                    has_file_parts = True
                    aid = str(part.get("attachment_id", ""))
                    if aid in url_map:
                        url, dl_url, exp = url_map[aid]
                        part["url"] = url
                        part["download_url"] = dl_url
                        part["downloadUrl"] = dl_url
                        if exp:
                            part["url_expires_at"] = exp.isoformat()
                            part["urlExpiresAt"] = exp.isoformat()
                        if aid in att_map:
                            meta = att_map[aid]
                            part["bucket"] = meta.bucket
                            part["object_name"] = meta.object_name
                            part["objectName"] = meta.object_name
                            part["contentType"] = meta.mime_type
                    elif part.get("storage_uri"):
                        url, exp = await self.generate_attachment_url(part["storage_uri"], expires_in=expires_in)
                        part["url"] = url
                        if exp:
                            part["url_expires_at"] = exp.isoformat()
                            part["urlExpiresAt"] = exp.isoformat()

            if not has_file_parts and getattr(msg, "attachment_ids", None):
                for aid in msg.attachment_ids:
                    s_aid = str(aid)
                    if s_aid in att_map:
                        meta = att_map[s_aid]
                        url, dl_url, exp = url_map.get(s_aid, (None, None, None))
                        part_dict: dict[str, Any] = {
                            "type": "file",
                            "attachment_id": str(meta.id),
                            "url": url,
                            "download_url": dl_url,
                            "downloadUrl": dl_url,
                            "filename": meta.filename,
                            "contentType": meta.mime_type,
                            "mediaType": meta.mime_type,
                            "size": meta.size,
                            "bucket": meta.bucket,
                            "object_name": meta.object_name,
                            "objectName": meta.object_name,
                            "storage_uri": meta.storage_uri,
                        }
                        if exp:
                            part_dict["url_expires_at"] = exp.isoformat()
                            part_dict["urlExpiresAt"] = exp.isoformat()
                        msg_parts.append(part_dict)

            msg.parts = msg_parts
