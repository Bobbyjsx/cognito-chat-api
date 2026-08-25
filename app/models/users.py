from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

from pydantic import BaseModel, EmailStr, Field


def _next_6h_reset() -> datetime:
    return datetime.now(timezone.utc) + timedelta(hours=6)


def _next_weekly_reset() -> datetime:
    return datetime.now(timezone.utc) + timedelta(weeks=1)


class UserDB(BaseModel):
    """Pydantic model representing a User document in Firestore."""

    id: UUID | str = Field(default_factory=uuid4)
    email: str
    hashed_password: str

    # Lifetime / all-time cumulative tokens used across account history
    tokens_used: int = 0

    # 6-hourly rolling window
    tokens_used_6h: int = 0
    token_limit_6h: int | None = None
    reset_at: datetime = Field(default_factory=_next_6h_reset)

    # Weekly rolling window
    tokens_used_weekly: int = 0
    token_limit_weekly: int | None = None
    weekly_reset_at: datetime = Field(default_factory=_next_weekly_reset)

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class UserCreate(BaseModel):
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: UUID | str
    email: str
    tokens_used: int
    tokens_used_6h: int
    token_limit_6h: int
    reset_at: datetime
    pct_6h: float = 0.0
    reset_countdown_6h: str = "Resets soon"
    tokens_used_weekly: int
    token_limit_weekly: int
    weekly_reset_at: datetime
    pct_weekly: float = 0.0
    reset_countdown_weekly: str = "Resets soon"


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class PasswordResetRequest(BaseModel):
    email: EmailStr
    new_password: str


class RefreshRequest(BaseModel):
    refresh_token: str
