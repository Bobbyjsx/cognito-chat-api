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
    # Not all models support all reasoning levels - this reflects actual Gemini capabilities
    model_reasoning_modes: dict[str, list[str]] = Field(
        default_factory=lambda: {
            # Full thinking model - supports all levels
            "gemini-3.6-flash": ["none", "minimal", "low", "medium", "high"],
            # Flash models - supports thinking but not ultra-low
            "gemini-3.5-flash": ["none", "low", "medium", "high"],
            # Lite model - basic reasoning only
            "gemini-3.5-flash-lite": ["none", "minimal"],
            # Pro preview - deep reasoning, all levels
            "gemini-3.1-pro-preview": ["none", "minimal", "low", "medium", "high"],
            # Flash lite - no thinking support
            "gemini-3.1-flash-lite": ["none"],
            # Flash preview - supports thinking
            "gemini-3-flash-preview": ["none", "low", "medium", "high"],
        }
    )

    # Human-readable descriptions for each text model
    model_descriptions: dict[str, str] = Field(
        default_factory=lambda: {
            "gemini-3.6-flash": "Latest Flash model with full thinking support and highest intelligence",
            "gemini-3.5-flash": "Fast and capable model with thinking modes for complex tasks",
            "gemini-3.5-flash-lite": "Ultra-fast lightweight model, best for simple and quick queries",
            "gemini-3.1-pro-preview": "Advanced Pro model with deep reasoning for complex code and analysis",
            "gemini-3.1-flash-lite": "Minimal-latency compact model, no thinking overhead",
            "gemini-3-flash-preview": "Balanced preview model with reasoning options",
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
