from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

SESSION_LIST_PREVIEW_CHARS = 280


def clip_session_preview(content: str | None) -> str | None:
    if not isinstance(content, str) or len(content) <= SESSION_LIST_PREVIEW_CHARS:
        return content
    return content[:SESSION_LIST_PREVIEW_CHARS].rstrip() + "…"


class MessageRole(str, Enum):
    """Supported roles in a chat conversation."""

    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"
    TOOL = "tool"


class ReadStatus(str, Enum):
    READ = "read"
    NOT_READ = "not read"


class GenerationStatus(str, Enum):
    QUEUED = "queued"
    RUNNING_LIVE = "running_live"
    RUNNING_WORKER = "running_worker"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class GenerationDB(BaseModel):
    """Pydantic model representing a durable generation execution."""

    id: UUID = Field(default_factory=uuid4)
    user_id: UUID | str
    session_id: UUID
    message_id: UUID | None = None
    user_message_id: UUID | None = None
    prompt: str | None = None
    status: GenerationStatus = GenerationStatus.QUEUED
    requested_model: str | None = None
    resolved_model: str | None = None
    requested_reasoning: str | None = None
    resolved_reasoning: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    error: str | None = None
    usage_tokens: int = 0
    buffered_text: str = ""
    buffered_thoughts: str = ""


class ChatMessageDB(BaseModel):
    """Pydantic model representing a Chat Message document in Firestore."""

    id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    role: MessageRole = MessageRole.USER
    content: str
    error: str | None = None
    attachment_ids: list[str] = Field(default_factory=list)
    generation_id: UUID | None = None
    parts: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ChatSessionDB(BaseModel):
    """Pydantic model representing a Chat Session document in Firestore."""

    id: UUID = Field(default_factory=uuid4)
    user_id: UUID | str
    title: str | None = None
    is_deleted: bool = False
    exclude_from_memory: bool = False
    share_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_message_content: str | None = None
    last_message_role: MessageRole | None = None
    read_status: ReadStatus = ReadStatus.READ
    messages: list[ChatMessageDB] = Field(default_factory=list)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=32_000)
    model: str | None = None
    reasoning: str | None = None
    routing_mode: str | None = None
    attachments: list[UUID] = Field(default_factory=list)


class ChatResponse(BaseModel):
    session_id: UUID
    title: str | None = None
    response: str
    model: str | None = None
    reasoning: str | None = None


class MessageSchema(BaseModel):
    id: UUID | None = None
    role: MessageRole
    content: str
    error: str | None = None
    attachment_ids: list[str] = Field(default_factory=list)
    parts: list[dict[str, Any]] = Field(default_factory=list)


class ChatSessionSchema(BaseModel):
    id: UUID
    user_id: UUID | str
    title: str | None = None
    messages: list[MessageSchema] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
    last_message_content: str | None = None
    last_message_role: str | None = None
    read_status: str = "read"
    active_generation_id: UUID | str | None = None
    exclude_from_memory: bool = False
    share_id: str | None = None


class ChatSessionListSchema(BaseModel):
    id: UUID
    user_id: UUID | str
    title: str | None = None
    created_at: datetime
    updated_at: datetime
    last_message_content: str | None = None
    last_message_role: str | None = None
    read_status: str = "read"
    active_generation_id: UUID | str | None = None
    exclude_from_memory: bool = False
    share_id: str | None = None


class CreateSharedChatRequest(BaseModel):
    title: str | None = None
    show_name: bool = True


class SharedChatMessageDB(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    role: MessageRole
    content: str
    attachment_ids: list[str] = Field(default_factory=list)
    parts: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SharedChatDB(BaseModel):
    id: str
    session_id: UUID
    user_id: UUID | str
    title: str | None = None
    show_name: bool = True
    author_name: str | None = None
    revoked_at: datetime | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    message_count: int = 0
    messages: list[SharedChatMessageDB] = Field(default_factory=list)


class SharedChatResponse(BaseModel):
    id: str
    session_id: UUID
    title: str | None = None
    author_name: str | None = None
    is_owner: bool = False
    created_at: datetime
    updated_at: datetime
    message_count: int
    messages: list[SharedChatMessageDB] = Field(default_factory=list)


class CreateSharedChatResponse(BaseModel):
    share_id: str
    session_id: UUID
    title: str | None = None
    author_name: str | None = None
    created_at: datetime
    message_count: int


class ContinueChatResponse(BaseModel):
    session_id: UUID
    title: str | None = None
    exclude_from_memory: bool = True
