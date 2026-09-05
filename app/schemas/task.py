from pydantic import BaseModel, Field


class GenerationTaskPayload(BaseModel):
    """Payload dispatched to Cloud Tasks and consumed by the Generation Worker."""

    generation_id: str = Field(..., description="Firestore generation document ID")
    attempt_number: int = Field(default=1, ge=1, description="Current execution attempt sequence")


class TaskExecutionResponse(BaseModel):
    """Result returned by the Cloud Tasks worker endpoint."""

    status: str
    generation_id: str
    attempt_number: int
    duration_ms: float


class TitleTaskPayload(BaseModel):
    """Payload dispatched to Cloud Tasks or background worker for AI title generation."""

    session_id: str = Field(..., description="Chat session ID")
    user_id: str = Field(..., description="User ID owning the session")
    prompt: str = Field(..., description="Prompt snippet (limited characters, text only, no attachments)")
    model: str | None = Field(default=None, description="Optional target model")
    attempt_number: int = Field(default=1, ge=1)


class TitleTaskExecutionResponse(BaseModel):
    """Result returned by the title generation worker."""

    status: str
    session_id: str
    title: str | None = None
    duration_ms: float
