from datetime import datetime, timezone

from pydantic import BaseModel, Field


class AppConfigDB(BaseModel):
    """Pydantic model representing global system configuration stored in Firestore."""

    id: str = "app_config"

    # Default Token Quota Configurations
    default_token_limit_6h: int = 60_000
    default_token_limit_weekly: int = 300_000

    # Feature Toggles
    enable_text_generation: bool = True
    enable_image_generation: bool = False
    enable_video_generation: bool = False

    # Text model configurations
    allowed_text_models: list[str] = Field(
        default_factory=lambda: [
            "gemini-3.6-flash",
            "gemini-3.5-flash",
            "gemini-3.5-flash-lite",
            "gemini-3.1-pro-preview",
            "gemini-3.1-flash-lite",
            "gemini-3-flash-preview",
        ]
    )
    default_text_model: str = "gemini-3.6-flash"

    # Reasoning / thinking effort configurations (Global source of truth)
    allowed_reasoning_levels: list[str] = Field(default_factory=lambda: ["none", "minimal", "low", "medium", "high"])
    default_reasoning_level: str = "medium"

    # Per-model reasoning mode mappings (filtered against allowed_reasoning_levels)
    model_reasoning_modes: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "gemini-3.6-flash": ["none", "minimal", "low", "medium", "high"],
            "gemini-3.5-flash": ["none", "minimal", "low", "medium"],
            "gemini-3.5-flash-lite": ["none"],
            "gemini-3.1-pro-preview": ["none", "minimal", "low", "medium", "high"],
            "gemini-3.1-flash-lite": ["none"],
            "gemini-3-flash-preview": ["none", "minimal", "low", "medium"],
        }
    )

    # Image model configurations
    allowed_image_models: list[str] = Field(
        default_factory=lambda: [
            "imagen-3.0-generate-002",
            "imagen-3.0-generate-001",
            "imagen-3.0-fast-generate-001",
        ]
    )

    # Video model configurations
    allowed_video_models: list[str] = Field(
        default_factory=lambda: [
            "veo-2.0-generate-001",
            "veo-1.0-generate-001",
            "veo-1.0-fast-generate-001",
        ]
    )

    # Tool configurations
    allowed_tools: list[str] = Field(default_factory=lambda: ["google_search", "code_execution"])

    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
