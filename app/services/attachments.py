"""Attachment service.

Owns the full attachment lifecycle: validation, MIME detection, upload to the
storage backend, metadata persistence, content preparation (Gemini parts) and
cleanup. The chat service never touches storage or upload mechanics — it asks
this service for prepared parts.
"""

from __future__ import annotations

import html
import io
import logging
import re
import zipfile
from uuid import UUID
from xml.etree import ElementTree

from fastapi import HTTPException

from app.models.attachments import AttachmentMetadata, AttachmentSchema
from app.models.config import AppConfigDB
from app.models.users import UserDB
from app.providers.base import BaseProvider
from app.repositories.attachments import AttachmentRepository
from app.storage.base import StorageBackend
from app.utils.mime import classify_attachment, detect_mime, is_textual

logger = logging.getLogger(__name__)

TEXT_PART_MAX_CHARS = 100_000

_DOCX_NAMESPACE = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
_XLSX_NAMESPACE = {
    "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


class AttachmentService:
    def __init__(self, repo: AttachmentRepository, storage: StorageBackend, provider: BaseProvider):
        self.repo = repo
        self.storage = storage
        self.provider = provider

    # ── ingest ────────────────────────────────────────────────────────────────

    async def ingest(
        self,
        user: UserDB,
        session_id: UUID | None,
        filename: str,
        content_type: str | None,
        data: bytes,
        config: AppConfigDB,
    ) -> AttachmentSchema:
        if not config.enable_attachments:
            raise HTTPException(status_code=403, detail="Attachments are currently disabled by admin.")

        if not filename or not filename.strip():
            raise HTTPException(status_code=400, detail="Attachment filename is required.")

        size = len(data)
        if size > config.attachment_max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File is too large. Maximum size is {config.attachment_max_size} bytes.",
            )

        mime = detect_mime(filename, content_type, data)
        attachment_type = classify_attachment(filename, mime)
        if attachment_type.value not in config.attachment_allowed_types:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Attachment type '{attachment_type.value}' is not allowed. "
                    f"Allowed types: {config.attachment_allowed_types}"
                ),
            )

        key = f"attachments/{user.id}/{attachment_type.value}/{filename}"
        storage_uri = await self.storage.upload_bytes(key, data, mime)

        metadata = AttachmentMetadata(
            user_id=user.id,
            session_id=session_id,
            filename=filename,
            mime_type=mime,
            size=size,
            storage_uri=storage_uri,
            type=attachment_type,
        )
        await self.repo.create(metadata)
        logger.info(
            "Attachment stored user=%s id=%s type=%s size=%d",
            user.id,
            metadata.id,
            attachment_type.value,
            size,
        )
        return AttachmentSchema.model_validate(metadata, from_attributes=True)

    # ── reads ─────────────────────────────────────────────────────────────────

    async def resolve_many(self, user_id: UUID, ids: list[UUID]) -> list[AttachmentMetadata]:
        if not ids:
            return []
        return await self.repo.get_many(user_id, ids)

    async def bind_session(self, metadata: AttachmentMetadata, session_id: UUID) -> None:
        """Associate an attachment with a session (persisted)."""
        if metadata.session_id is None or str(metadata.session_id) != str(session_id):
            metadata.session_id = session_id
            await self.repo.update_session(metadata.id, session_id)

    async def read_bytes(self, metadata: AttachmentMetadata) -> bytes:
        if not metadata.storage_uri:
            raise HTTPException(status_code=404, detail="Attachment has no stored content.")
        return await self.storage.read_bytes(metadata.storage_uri)

    # ── part preparation ──────────────────────────────────────────────────────

    async def prepare_parts(self, metadata: AttachmentMetadata) -> list[dict]:
        """Return provider-agnostic parts for an attachment.

        Textual attachments are converted to text parts here; media
        (image/pdf/audio/video) is delegated to the provider, which may upload
        to a provider File API and persist the resulting URI back onto
        ``metadata``.
        """
        if is_textual(metadata.type):
            data = await self.read_bytes(metadata)
            return [{"text": self._extract_text(metadata, data)}]

        data = await self.read_bytes(metadata)
        previous_uri = metadata.gemini_file_uri
        parts = await self.provider.parts_for_attachment(metadata, data)
        if metadata.gemini_file_uri != previous_uri:
            await self.repo.update_gemini_uri(metadata.id, metadata.gemini_file_uri)
        return parts

    # ── delete ────────────────────────────────────────────────────────────────

    async def delete(self, user_id: UUID, attachment_id: UUID) -> bool:
        metadata = await self.repo.get(attachment_id, user_id)
        if metadata is None:
            return False
        if metadata.storage_uri:
            try:
                await self.storage.delete(metadata.storage_uri)
            except Exception:
                logger.exception("Failed to delete object %s", metadata.storage_uri)
        await self.repo.delete(attachment_id)
        return True

    # ── text extraction ───────────────────────────────────────────────────────

    def _extract_text(self, metadata: AttachmentMetadata, data: bytes) -> str:
        try:
            if metadata.filename.lower().endswith(".docx"):
                text = _extract_docx_text(data)
            elif metadata.filename.lower().endswith(".xlsx"):
                text = _extract_xlsx_text(data)
            else:
                text = data.decode("utf-8", errors="replace")
        except Exception:
            logger.exception("Text extraction failed for %s", metadata.filename)
            text = data.decode("utf-8", errors="replace")

        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text).strip()
        if len(text) > TEXT_PART_MAX_CHARS:
            text = text[:TEXT_PART_MAX_CHARS]
            logger.warning("Truncated text attachment %s to %d chars", metadata.filename, TEXT_PART_MAX_CHARS)
        return text


def _extract_docx_text(data: bytes) -> str:
    """Extract text from a .docx (a zip of XML parts) without extra deps."""
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        xml = archive.read("word/document.xml").decode("utf-8", errors="ignore")

    xml = re.sub(r"<w:tab[^>]*/>", "\t", xml)
    xml = re.sub(r"<w:br[^>]*/>", "\n", xml)
    xml = re.sub(r"</w:p>", "\n", xml)
    text = re.sub(r"<[^>]+>", "", xml)
    return html.unescape(text)


def _extract_xlsx_text(data: bytes) -> str:
    """Extract cell text from the first worksheet of an .xlsx file."""
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()

        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in names:
            root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            for si in root.findall("m:si", _XLSX_NAMESPACE):
                shared_strings.append("".join(t.text or "" for t in si.findall(".//m:t", _XLSX_NAMESPACE)))

        sheets = sorted(
            n
            for n in names
            if n.startswith("xl/worksheets/") and n.endswith(".xml")
        )
        if not sheets:
            return ""
        root = ElementTree.fromstring(archive.read(sheets[0]))

    rows: list[str] = []
    for row in root.findall(".//m:row", _XLSX_NAMESPACE):
        cells: list[str] = []
        for cell in row.findall("m:c", _XLSX_NAMESPACE):
            cell_type = cell.get("t")
            value = cell.find("m:v", _XLSX_NAMESPACE)
            raw = value.text if value is not None and value.text else ""
            if cell_type == "s" and raw:
                try:
                    raw = shared_strings[int(raw)]
                except (ValueError, IndexError):
                    raw = ""
            cells.append(raw)
        rows.append(",".join(cells))
    return "\n".join(rows)
