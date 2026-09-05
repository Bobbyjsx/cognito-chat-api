from datetime import datetime, timezone
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, computed_field, model_validator


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
    storage backend (e.g. Google Cloud Storage) referenced by ``bucket`` and ``object_name``
    (or ``storage_uri``).
    """

    id: UUID | str = Field(default_factory=uuid4)
    user_id: UUID | str
    session_id: UUID | str | None = None
    filename: str
    mime_type: str
    size: int = 0
    bucket: str | None = None
    object_name: str | None = None
    storage_uri: str | None = None
    type: AttachmentType = AttachmentType.document
    # Cache of the provider-side file URI (Gemini Files API) so large media is
    # uploaded only once per attachment.
    gemini_file_uri: str | None = None
    is_temporary: bool = True
    uploaded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    deleted_at: datetime | None = None

    @model_validator(mode="after")
    def populate_storage_identity(self) -> "AttachmentMetadata":
        if self.storage_uri and (not self.bucket or not self.object_name):
            if self.storage_uri.startswith("gs://"):
                parts = self.storage_uri[5:].split("/", 1)
                if len(parts) == 2:
                    if not self.bucket:
                        self.bucket = parts[0]
                    if not self.object_name:
                        self.object_name = parts[1]
            elif self.storage_uri.startswith("local://"):
                if not self.bucket:
                    self.bucket = "local"
                if not self.object_name:
                    self.object_name = self.storage_uri[8:]
        elif self.bucket and self.object_name and not self.storage_uri:
            if self.bucket == "local":
                self.storage_uri = f"local://{self.object_name}"
            else:
                self.storage_uri = f"gs://{self.bucket}/{self.object_name}"
        return self


class AttachmentSchema(BaseModel):
    """Wire representation of an attachment (no internal/provider fields)."""

    id: UUID
    filename: str
    mime_type: str
    size: int
    type: AttachmentType
    session_id: UUID | None = None
    bucket: str | None = None
    object_name: str | None = None
    storage_uri: str | None = None
    url: str | None = None
    download_url: str | None = None
    url_expires_at: datetime | None = None
    is_temporary: bool = True
    uploaded_at: datetime
    deleted_at: datetime | None = None

    @model_validator(mode="after")
    def populate_storage_identity(self) -> "AttachmentSchema":
        if self.storage_uri and (not self.bucket or not self.object_name):
            if self.storage_uri.startswith("gs://"):
                parts = self.storage_uri[5:].split("/", 1)
                if len(parts) == 2:
                    if not self.bucket:
                        self.bucket = parts[0]
                    if not self.object_name:
                        self.object_name = parts[1]
            elif self.storage_uri.startswith("local://"):
                if not self.bucket:
                    self.bucket = "local"
                if not self.object_name:
                    self.object_name = self.storage_uri[8:]
        elif self.bucket and self.object_name and not self.storage_uri:
            if self.bucket == "local":
                self.storage_uri = f"local://{self.object_name}"
            else:
                self.storage_uri = f"gs://{self.bucket}/{self.object_name}"
        return self

    @computed_field
    @property
    def content_type(self) -> str:
        return self.mime_type


class PresignedUploadRequest(BaseModel):
    filename: str
    mime_type: str
    size: int
    session_id: UUID | None = None
    is_temporary: bool = True


class PresignedUploadResponse(BaseModel):
    attachment_id: UUID
    upload_url: str
    method: str = "PUT"
    headers: dict[str, str] = Field(default_factory=dict)
    attachment: AttachmentSchema
