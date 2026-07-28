from datetime import datetime, timezone
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ChatMessageDB(BaseModel):
    """Pydantic model representing a Chat Message document in Firestore."""

    id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    role: str
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ChatSessionDB(BaseModel):
    """Pydantic model representing a Chat Session document in Firestore."""

    id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_message_content: str | None = None
    last_message_role: str | None = None
    read_status: str = "read"
    # We won't store messages directly inside the session document in Firestore,
    # they will be in a subcollection, but we can load them into this model in the repo.
    messages: list[ChatMessageDB] = Field(default_factory=list)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    session_id: UUID
    response: str


class MessageSchema(BaseModel):
    role: str
    content: str


class ChatSessionSchema(BaseModel):
    id: UUID
    user_id: UUID
    messages: list[MessageSchema] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
    last_message_content: str | None = None
    last_message_role: str | None = None
    read_status: str = "read"


class ChatSessionListSchema(BaseModel):
    id: UUID
    user_id: UUID
    created_at: datetime
    updated_at: datetime
    last_message_content: str | None = None
    last_message_role: str | None = None
    read_status: str = "read"
