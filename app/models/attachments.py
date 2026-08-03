from datetime import datetime, timezone
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class AttachmentType(str, Enum):
    """Categorization of an uploaded attachment.

    ``document`` covers text-ish files (txt, markdown, CSV, DOCX, source code),
    ``json`` and ``text`` are separate for clarity, ``spreadsheet`` covers
    Excel-like files. Designed for future expansion (e.g. ``archive``).
    """

    image = "image"
    pdf = "pdf"
    document = "document"
    audio = "audio"
    video = "video"
    spreadsheet = "spreadsheet"
    json = "json"
    text = "text"


class AttachmentMetadata(BaseModel):
    """Pydantic model representing an Attachment document in Firestore.

    Only metadata is stored in Firestore; the actual bytes live in an object
    storage backend (e.g. Google Cloud Storage) referenced by ``storage_uri``.
    """

    id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    session_id: UUID | None = None
    filename: str
    mime_type: str
    size: int = 0
    storage_uri: str | None = None
    type: AttachmentType = AttachmentType.document
    # Cache of the provider-side file URI (Gemini Files API) so large media is
    # uploaded only once per attachment.
    gemini_file_uri: str | None = None
    uploaded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AttachmentSchema(BaseModel):
    """Wire representation of an attachment (no internal/provider fields)."""

    id: UUID
    filename: str
    mime_type: str
    size: int
    type: AttachmentType
    session_id: UUID | None = None
    storage_uri: str | None = None
    uploaded_at: datetime
