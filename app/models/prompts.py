from datetime import datetime, timezone
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class PromptDB(BaseModel):
    """Pydantic model representing a Prompt document in Firestore."""

    id: UUID | str = Field(default_factory=uuid4)
    user_id: str | None = None
    title: str
    description: str
    category: str
    prompt: str
    tags: list[str] = Field(default_factory=list)
    is_custom: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PromptCreate(BaseModel):
    title: str
    description: str
    category: str = "custom"
    prompt: str
    tags: list[str] = Field(default_factory=list)


class PromptUpdate(BaseModel):
    title: str | None = None
    description: str | None = None
    category: str | None = None
    prompt: str | None = None
    tags: list[str] | None = None


class PromptResponse(BaseModel):
    id: UUID | str
    title: str
    description: str
    category: str
    prompt: str
    tags: list[str]
    isCustom: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        populate_by_name = True

        @staticmethod
        def alias_generator(x: str) -> str:
            return "isCustom" if x == "is_custom" else x
