from datetime import datetime, timezone
from uuid import UUID, uuid4

from pydantic import BaseModel, EmailStr, Field


class UserDB(BaseModel):
    """Pydantic model representing a User document in Firestore."""

    id: UUID = Field(default_factory=uuid4)
    email: EmailStr
    hashed_password: str
    tokens_used: int = 0
    token_limit: int = 50000
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class UserCreate(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: UUID
    email: EmailStr
    tokens_used: int
    token_limit: int


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class PasswordResetRequest(BaseModel):
    email: EmailStr
    new_password: str
