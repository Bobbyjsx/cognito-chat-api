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
