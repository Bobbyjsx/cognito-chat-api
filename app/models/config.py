from datetime import datetime, timezone
from pydantic import BaseModel, Field


class AppConfigDB(BaseModel):
    """Pydantic model representing global system configuration stored in Firestore."""

    id: str = "app_config"

    # Text model configurations
    allowed_text_models: list[str] = Field(
        default_factory=lambda: [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
        ]
    )
    default_text_model: str = "gemini-2.5-flash"

    # Reasoning / thinking effort configurations
    allowed_reasoning_levels: list[str] = Field(
        default_factory=lambda: ["none", "minimal", "low", "medium", "high"]
    )
    default_reasoning_level: str = "medium"

    # Tool and multimodal model configurations
    allowed_image_models: list[str] = Field(
        default_factory=lambda: ["imagen-3.0-generate-002"]
    )
    allowed_video_models: list[str] = Field(
        default_factory=lambda: ["veo-2.0-generate-001"]
    )
    allowed_tools: list[str] = Field(
        default_factory=lambda: ["google_search"]
    )

    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
