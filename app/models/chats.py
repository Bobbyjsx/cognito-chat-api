from datetime import datetime, timezone
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ReadStatus(str, Enum):
    READ = "read"
    UNREAD = "unread"


class ChatMessageDB(BaseModel):
    """Pydantic model representing a Chat Message document in Firestore."""

    id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    role: str
    content: str
    error: str | None = None
    attachment_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ChatSessionDB(BaseModel):
    """Pydantic model representing a Chat Session document in Firestore."""

    id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    title: str | None = None
    is_deleted: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_message_content: str | None = None
    last_message_role: str | None = None
    read_status: str = "read"
    messages: list[ChatMessageDB] = Field(default_factory=list)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=32_000)
    model: str | None = None
    reasoning: str | None = None
    attachments: list[UUID] = Field(default_factory=list)


class ChatResponse(BaseModel):
    session_id: UUID
    title: str | None = None
    response: str


class MessageSchema(BaseModel):
    role: str
    content: str
    error: str | None = None
    attachment_ids: list[str] = Field(default_factory=list)


class ChatSessionSchema(BaseModel):
    id: UUID
    user_id: UUID
    title: str | None = None
    messages: list[MessageSchema] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
    last_message_content: str | None = None
    last_message_role: str | None = None
    read_status: str = "read"


class ChatSessionListSchema(BaseModel):
    id: UUID
    user_id: UUID
    title: str | None = None
    created_at: datetime
    updated_at: datetime
    last_message_content: str | None = None
    last_message_role: str | None = None
    read_status: str = "read"
