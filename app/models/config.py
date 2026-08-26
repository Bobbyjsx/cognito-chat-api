from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, computed_field

from app.models.attachments import AttachmentType


class ReasoningLevel(str, Enum):
    """Unified thinking / reasoning effort levels."""

    FAST = "fast"
    BALANCED = "balanced"
    EXTENDED = "extended"


class ModelProvider(str, Enum):
    """Supported AI model providers."""

    GOOGLE = "google"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    MISTRAL = "mistral"
    META = "meta"
    OTHER = "other"


class ModelStatus(str, Enum):
    """Operational lifecycle status of a model."""

    ACTIVE = "active"
    DISABLED = "disabled"
    TEMPORARILY_UNAVAILABLE = "temporarily_unavailable"
    DEPRECATED = "deprecated"


class RoutingMode(str, Enum):
    """Unified routing optimization strategy modes."""

    FAST = "fast"
    BALANCED = "balanced"
    EXTENDED = "extended"
    CUSTOM = "custom"


def normalize_reasoning_level(val: str | ReasoningLevel | None) -> ReasoningLevel | None:
    """Normalize any raw reasoning or policy string into canonical ReasoningLevel."""
    if val is None:
        return None
    raw = (val.value if isinstance(val, ReasoningLevel) else str(val)).lower().strip()
    if raw in ("fast", "none", "minimal", "low", "speed"):
        return ReasoningLevel.FAST
    if raw in ("balanced", "medium"):
        return ReasoningLevel.BALANCED
    if raw in ("extended", "high", "quality", "deep"):
        return ReasoningLevel.EXTENDED
    return None


class ToolName(str, Enum):
    """Available runtime tools."""

    GOOGLE_SEARCH = "google_search"
    CODE_EXECUTION = "code_execution"


class TextModelConfig(BaseModel):
    """Configuration and routing metadata for a single text model."""

    description: str
    enabled: bool = True
    # Reasoning modes this model supports. Must be a subset of the global
    # allowed_reasoning_levels. If a level appears here but NOT in the
    # global list, the global list takes precedence (it won't be offered).
    reasoning_modes: list[ReasoningLevel]

    # ── Routing Scores (Scale: 0.0 - 1.0) ─────────────────────────────────────
    complexity_score: float = Field(default=0.5, ge=0.0, le=1.0)
    reasoning_score: float = Field(default=0.5, ge=0.0, le=1.0)
    coding_score: float = Field(default=0.5, ge=0.0, le=1.0)
    creative_score: float = Field(default=0.5, ge=0.0, le=1.0)
    context_score: float = Field(default=0.5, ge=0.0, le=1.0)
    vision_score: float = Field(default=0.5, ge=0.0, le=1.0)
    tool_calling_score: float = Field(default=0.5, ge=0.0, le=1.0)
    structured_output_score: float = Field(default=0.5, ge=0.0, le=1.0)
    speed_score: float = Field(default=0.5, ge=0.0, le=1.0)
    quality_score: float = Field(default=0.5, ge=0.0, le=1.0)

    # ── Economics & Limits ───────────────────────────────────────────────────
    input_cost_per_million: float = Field(default=0.10, ge=0.0)
    output_cost_per_million: float = Field(default=0.40, ge=0.0)
    context_window_tokens: int = Field(default=1_000_000, ge=1)

    # ── Modality & Feature Capabilities ───────────────────────────────────────
    supports_vision: bool = True
    supports_tools: bool = True
    supports_structured_output: bool = True
    supports_audio: bool = False
    supports_web_search: bool = True
    supports_code_execution: bool = True

    # ── Operational State ────────────────────────────────────────────────────
    provider: ModelProvider = ModelProvider.GOOGLE
    status: ModelStatus = ModelStatus.ACTIVE


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
    # When True: use backend AI model for STT (always show mic).
    # When False: use browser Web Speech API (mic shown only on supported browsers).
    enable_ai_stt: bool = False
    # Gemini model used for AI STT transcription (cheapest audio-capable option).
    # gemini-3.1-flash-lite natively accepts audio input ($0.50/1M tokens).
    # Cheaper flash-lite tiers (e.g. gemini-2.0-flash-lite) do NOT support audio.
    stt_model: str = "gemini-3.1-flash-lite"

    # ── Smart Model Router Config ─────────────────────────────────────────────
    enable_smart_routing: bool = True
    router_model: str = "gemini-3.1-flash-lite"
    default_routing_mode: RoutingMode = RoutingMode.BALANCED

    # ── Reasoning (Global Source of Truth) ────────────────────────────────────
    # If a mode is listed on a model but NOT here, it is silently ignored.
    allowed_reasoning_levels: list[ReasoningLevel] = Field(
        default_factory=lambda: [
            ReasoningLevel.FAST,
            ReasoningLevel.BALANCED,
            ReasoningLevel.EXTENDED,
        ]
    )
    default_reasoning_level: ReasoningLevel = ReasoningLevel.BALANCED
    default_text_model: str = "gemini-3.6-flash"

    # ── Structured Text Models ─────────────────────────────────────────────────
    # Single source of truth for model availability, descriptions, capabilities,
    # and routing parameters.
    models_list: dict[str, TextModelConfig] = Field(
        default_factory=lambda: {
            "auto": TextModelConfig(
                description="Automatically selects the optimal model based on prompt complexity and requirements",
                enabled=True,
                reasoning_modes=[
                    ReasoningLevel.FAST,
                    ReasoningLevel.BALANCED,
                    ReasoningLevel.EXTENDED,
                ],
                complexity_score=1.0,
                reasoning_score=1.0,
                coding_score=1.0,
                creative_score=1.0,
                context_score=1.0,
                vision_score=1.0,
                tool_calling_score=1.0,
                structured_output_score=1.0,
                speed_score=1.0,
                quality_score=1.0,
                input_cost_per_million=0.0,
                output_cost_per_million=0.0,
                context_window_tokens=2_000_000,
                supports_vision=True,
                supports_tools=True,
                supports_structured_output=True,
                supports_audio=True,
                supports_web_search=True,
                supports_code_execution=True,
                provider=ModelProvider.GOOGLE,
                status=ModelStatus.ACTIVE,
            ),
            "gemini-3.6-flash": TextModelConfig(
                description="Latest Flash model with full thinking support and highest intelligence",
                enabled=True,
                reasoning_modes=[
                    ReasoningLevel.FAST,
                    ReasoningLevel.BALANCED,
                    ReasoningLevel.EXTENDED,
                ],
                complexity_score=0.85,
                reasoning_score=0.88,
                coding_score=0.88,
                creative_score=0.85,
                context_score=0.90,
                vision_score=0.90,
                tool_calling_score=0.92,
                structured_output_score=0.90,
                speed_score=0.85,
                quality_score=0.88,
                input_cost_per_million=0.15,
                output_cost_per_million=0.60,
                context_window_tokens=1_000_000,
                supports_vision=True,
                supports_tools=True,
                supports_structured_output=True,
                supports_audio=True,
                supports_web_search=True,
                supports_code_execution=True,
                provider=ModelProvider.GOOGLE,
                status=ModelStatus.ACTIVE,
            ),
            "gemini-3.5-flash": TextModelConfig(
                description="Fast and capable model with thinking modes for complex tasks",
                enabled=True,
                reasoning_modes=[
                    ReasoningLevel.FAST,
                    ReasoningLevel.BALANCED,
                    ReasoningLevel.EXTENDED,
                ],
                complexity_score=0.75,
                reasoning_score=0.75,
                coding_score=0.78,
                creative_score=0.80,
                context_score=0.85,
                vision_score=0.85,
                tool_calling_score=0.85,
                structured_output_score=0.85,
                speed_score=0.90,
                quality_score=0.80,
                input_cost_per_million=0.075,
                output_cost_per_million=0.30,
                context_window_tokens=1_000_000,
                supports_vision=True,
                supports_tools=True,
                supports_structured_output=True,
                supports_audio=True,
                supports_web_search=True,
                supports_code_execution=True,
                provider=ModelProvider.GOOGLE,
                status=ModelStatus.ACTIVE,
            ),
            "gemini-3.5-flash-lite": TextModelConfig(
                description="Ultra-fast lightweight model, best for simple and quick queries",
                enabled=True,
                reasoning_modes=[ReasoningLevel.FAST],
                complexity_score=0.40,
                reasoning_score=0.35,
                coding_score=0.50,
                creative_score=0.50,
                context_score=0.80,
                vision_score=0.80,
                tool_calling_score=0.80,
                structured_output_score=0.80,
                speed_score=0.98,
                quality_score=0.65,
                input_cost_per_million=0.0375,
                output_cost_per_million=0.15,
                context_window_tokens=1_000_000,
                supports_vision=True,
                supports_tools=True,
                supports_structured_output=True,
                supports_audio=False,
                supports_web_search=True,
                supports_code_execution=True,
                provider=ModelProvider.GOOGLE,
                status=ModelStatus.ACTIVE,
            ),
            "gemini-3.1-pro-preview": TextModelConfig(
                description="Advanced Pro model with deep reasoning for complex code and analysis",
                enabled=True,
                reasoning_modes=[
                    ReasoningLevel.FAST,
                    ReasoningLevel.BALANCED,
                    ReasoningLevel.EXTENDED,
                ],
                complexity_score=0.95,
                reasoning_score=0.98,
                coding_score=0.95,
                creative_score=0.92,
                context_score=0.98,
                vision_score=0.95,
                tool_calling_score=0.95,
                structured_output_score=0.95,
                speed_score=0.60,
                quality_score=0.98,
                input_cost_per_million=1.25,
                output_cost_per_million=5.00,
                context_window_tokens=2_000_000,
                supports_vision=True,
                supports_tools=True,
                supports_structured_output=True,
                supports_audio=True,
                supports_web_search=True,
                supports_code_execution=True,
                provider=ModelProvider.GOOGLE,
                status=ModelStatus.ACTIVE,
            ),
            "gemini-3.1-flash-lite": TextModelConfig(
                description="Minimal-latency compact model, no thinking overhead",
                enabled=True,
                reasoning_modes=[ReasoningLevel.FAST],
                complexity_score=0.25,
                reasoning_score=0.20,
                coding_score=0.35,
                creative_score=0.40,
                context_score=0.80,
                vision_score=0.80,
                tool_calling_score=0.75,
                structured_output_score=0.75,
                speed_score=1.00,
                quality_score=0.55,
                input_cost_per_million=0.025,
                output_cost_per_million=0.10,
                context_window_tokens=1_000_000,
                supports_vision=True,
                supports_tools=True,
                supports_structured_output=True,
                supports_audio=True,
                supports_web_search=True,
                supports_code_execution=True,
                provider=ModelProvider.GOOGLE,
                status=ModelStatus.ACTIVE,
            ),
            "gemini-3-flash-preview": TextModelConfig(
                description="Balanced preview model with reasoning options",
                enabled=True,
                reasoning_modes=[
                    ReasoningLevel.FAST,
                    ReasoningLevel.BALANCED,
                    ReasoningLevel.EXTENDED,
                ],
                complexity_score=0.70,
                reasoning_score=0.70,
                coding_score=0.72,
                creative_score=0.75,
                context_score=0.85,
                vision_score=0.85,
                tool_calling_score=0.82,
                structured_output_score=0.82,
                speed_score=0.88,
                quality_score=0.75,
                input_cost_per_million=0.075,
                output_cost_per_million=0.30,
                context_window_tokens=1_000_000,
                supports_vision=True,
                supports_tools=True,
                supports_structured_output=True,
                supports_audio=True,
                supports_web_search=True,
                supports_code_execution=True,
                provider=ModelProvider.GOOGLE,
                status=ModelStatus.ACTIVE,
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
    allowed_tools: list[ToolName] = Field(
        default_factory=lambda: [
            ToolName.GOOGLE_SEARCH,
            ToolName.CODE_EXECUTION,
        ]
    )

    # ── Attachments ────────────────────────────────────────────────────────────
    # Master switch for the attachment pipeline (upload endpoint + parts in chat).
    enable_attachments: bool = True
    # Maximum size of a single uploaded file, in bytes (default 20 MB).
    attachment_max_size: int = 20_000_000
    # Maximum number of attachments allowed per chat message.
    attachment_max_count: int = 10
    # Attachment types allowed to be uploaded, as AttachmentType values.
    attachment_allowed_types: list[AttachmentType] = Field(
        default_factory=lambda: [
            AttachmentType.image,
            AttachmentType.pdf,
            AttachmentType.document,
            AttachmentType.audio,
            AttachmentType.video,
            AttachmentType.spreadsheet,
            AttachmentType.json,
            AttachmentType.text,
        ]
    )

    # ── Context Management ─────────────────────────────────────────────────────
    # Trim the conversation history sent to the model instead of sending it all.
    context_trim_enabled: bool = True
    # Approximate token budget for the retained history.
    context_max_tokens: int = 400_000
    # Always keep at least this many most-recent messages, even over budget.
    context_keep_recent: int = 6

    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # ── Computed helpers (not stored, derived at runtime) ─────────────────────

    @computed_field
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
