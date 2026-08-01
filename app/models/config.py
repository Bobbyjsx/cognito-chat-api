from datetime import datetime, timezone

from pydantic import BaseModel, Field


class TextModelConfig(BaseModel):
    """Configuration for a single text model."""

    description: str
    enabled: bool = True
    # Reasoning modes this model supports. Must be a subset of the global
    # allowed_reasoning_levels. If a level appears here but NOT in the
    # global list, the global list takes precedence (it won't be offered).
    reasoning_modes: list[str]


class AppConfigDB(BaseModel):
    """Pydantic model representing global system configuration stored in Firestore."""

    id: str = "app_config"

    # ── Token Quota ────────────────────────────────────────────────────────────
    default_token_limit_6h: int = 60_000
    default_token_limit_weekly: int = 300_000

    # ── Feature Toggles ────────────────────────────────────────────────────────
    enable_text_generation: bool = True
    enable_image_generation: bool = False
    enable_video_generation: bool = False

    # ── Reasoning (Global Source of Truth) ────────────────────────────────────
    # If a mode is listed on a model but NOT here, it is silently ignored.
    allowed_reasoning_levels: list[str] = Field(
        default_factory=lambda: ["none", "minimal", "low", "medium", "high"]
    )
    default_reasoning_level: str = "medium"
    default_text_model: str = "gemini-3.6-flash"

    # ── Structured Text Models ─────────────────────────────────────────────────
    # Single source of truth for model availability, descriptions, and
    # per-model reasoning support.  Replaces the old flat lists:
    #   - allowed_text_models  → derived as [k for k,v in models_list if v.enabled]
    #   - model_reasoning_modes → moved into each model's reasoning_modes field
    #   - model_descriptions    → moved into each model's description field
    models_list: dict[str, TextModelConfig] = Field(
        default_factory=lambda: {
            "gemini-3.6-flash": TextModelConfig(
                description="Latest Flash model with full thinking support and highest intelligence",
                enabled=True,
                reasoning_modes=["none", "minimal", "low", "medium", "high"],
            ),
            "gemini-3.5-flash": TextModelConfig(
                description="Fast and capable model with thinking modes for complex tasks",
                enabled=True,
                reasoning_modes=["none", "low", "medium", "high"],
            ),
            "gemini-3.5-flash-lite": TextModelConfig(
                description="Ultra-fast lightweight model, best for simple and quick queries",
                enabled=True,
                reasoning_modes=["none", "minimal"],
            ),
            "gemini-3.1-pro-preview": TextModelConfig(
                description="Advanced Pro model with deep reasoning for complex code and analysis",
                enabled=True,
                reasoning_modes=["none", "minimal", "low", "medium", "high"],
            ),
            "gemini-3.1-flash-lite": TextModelConfig(
                description="Minimal-latency compact model, no thinking overhead",
                enabled=True,
                reasoning_modes=["none"],
            ),
            "gemini-3-flash-preview": TextModelConfig(
                description="Balanced preview model with reasoning options",
                enabled=True,
                reasoning_modes=["none", "low", "medium", "high"],
            ),
        }
    )

    # ── Image / Video / Tool configs ───────────────────────────────────────────
    allowed_image_models: list[str] = Field(
        default_factory=lambda: [
            "imagen-3.0-generate-002",
            "imagen-3.0-generate-001",
            "imagen-3.0-fast-generate-001",
        ]
    )
    allowed_video_models: list[str] = Field(
        default_factory=lambda: [
            "veo-2.0-generate-001",
            "veo-1.0-generate-001",
            "veo-1.0-fast-generate-001",
        ]
    )
    allowed_tools: list[str] = Field(default_factory=lambda: ["google_search", "code_execution"])

    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # ── Computed helpers (not stored, derived at runtime) ─────────────────────

    @property
    def allowed_text_models(self) -> list[str]:
        """All enabled model names — derived from models_list."""
        return [name for name, cfg in self.models_list.items() if cfg.enabled]

    def get_reasoning_modes_for_model(self, model_name: str) -> list[str]:
        """Return the intersection of a model's reasoning_modes and the global
        allowed_reasoning_levels. Global list is always the authority."""
        model_cfg = self.models_list.get(model_name)
        if not model_cfg:
            return self.allowed_reasoning_levels
        return [m for m in model_cfg.reasoning_modes if m in self.allowed_reasoning_levels]
