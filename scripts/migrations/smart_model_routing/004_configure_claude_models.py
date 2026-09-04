import asyncio
import os
import sys
from datetime import datetime, timezone

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from app.database import create_db_client, init_db
from app.models.config import (
    ModelProvider,
    ModelStatus,
    ReasoningLevel,
    TextModelConfig,
)

CLAUDE_MODELS: dict[str, TextModelConfig] = {
    "claude-3-5-haiku": TextModelConfig(
        description="Ultra-fast lightweight Claude model for low-latency reasoning and efficient execution",
        enabled=True,
        reasoning_modes=[ReasoningLevel.FAST],
        complexity_score=0.60,
        reasoning_score=0.60,
        coding_score=0.70,
        creative_score=0.70,
        context_score=0.90,
        vision_score=0.85,
        tool_calling_score=0.85,
        structured_output_score=0.85,
        speed_score=0.96,
        quality_score=0.78,
        input_cost_per_million=0.80,
        output_cost_per_million=4.00,
        context_window_tokens=200_000,
        supports_vision=True,
        supports_tools=True,
        supports_structured_output=True,
        supports_audio=False,
        supports_web_search=True,
        supports_code_execution=False,
        provider=ModelProvider.ANTHROPIC,
        status=ModelStatus.ACTIVE,
    ),
    "claude-3-5-sonnet": TextModelConfig(
        description="High intelligence model with outstanding coding, analysis, and visual understanding",
        enabled=True,
        reasoning_modes=[
            ReasoningLevel.FAST,
            ReasoningLevel.BALANCED,
            ReasoningLevel.EXTENDED,
        ],
        complexity_score=0.92,
        reasoning_score=0.92,
        coding_score=0.94,
        creative_score=0.90,
        context_score=0.95,
        vision_score=0.95,
        tool_calling_score=0.94,
        structured_output_score=0.92,
        speed_score=0.78,
        quality_score=0.94,
        input_cost_per_million=3.00,
        output_cost_per_million=15.00,
        context_window_tokens=200_000,
        supports_vision=True,
        supports_tools=True,
        supports_structured_output=True,
        supports_audio=False,
        supports_web_search=True,
        supports_code_execution=False,
        provider=ModelProvider.ANTHROPIC,
        status=ModelStatus.ACTIVE,
    ),
    "claude-3-7-sonnet": TextModelConfig(
        description="Anthropic's hybrid model with dynamic reasoning/extended thinking and state-of-the-art coding",
        enabled=True,
        reasoning_modes=[
            ReasoningLevel.FAST,
            ReasoningLevel.BALANCED,
            ReasoningLevel.EXTENDED,
        ],
        complexity_score=0.98,
        reasoning_score=0.98,
        coding_score=0.98,
        creative_score=0.95,
        context_score=0.95,
        vision_score=0.95,
        tool_calling_score=0.96,
        structured_output_score=0.95,
        speed_score=0.70,
        quality_score=0.98,
        input_cost_per_million=3.00,
        output_cost_per_million=15.00,
        context_window_tokens=200_000,
        supports_vision=True,
        supports_tools=True,
        supports_structured_output=True,
        supports_audio=False,
        supports_web_search=True,
        supports_code_execution=False,
        provider=ModelProvider.ANTHROPIC,
        status=ModelStatus.ACTIVE,
    ),
}


async def migrate():
    """Update configs/app_config in Firestore to include exactly the 3 selected Claude models.

    Date: 2026-09-04
    Idempotent: Replaces or adds the 3 canonical Claude models, prunes extraneous
    Claude models, preserves all other models (e.g. Gemini/Google), and clears Redis cache.
    """
    print("  [004_configure_claude_models] Initializing Firestore connection...")
    init_db()
    db = create_db_client()

    config_ref = db.collection("configs").document("app_config")
    config_doc = await config_ref.get()

    if not config_doc.exists:
        print("  [004_configure_claude_models] Error: 'configs/app_config' does not exist.")
        return

    existing = config_doc.to_dict() or {}
    existing_models = existing.get("models_list", {})

    # Build updated models_list:
    # 1. Keep all non-Anthropic models (Google / Gemini / Auto)
    updated_models: dict = {}
    for model_name, cfg in existing_models.items():
        provider_val = cfg.get("provider") if isinstance(cfg, dict) else getattr(cfg, "provider", None)
        if provider_val != ModelProvider.ANTHROPIC.value and provider_val != "anthropic":
            updated_models[model_name] = cfg

    # 2. Add the 3 canonical Claude models
    for model_name, model_cfg in CLAUDE_MODELS.items():
        dumped = model_cfg.model_dump(mode="json")
        # Preserve user custom overrides if any existed previously (e.g. enabled toggle)
        if model_name in existing_models:
            old_enabled = existing_models[model_name].get("enabled")
            if old_enabled is not None:
                dumped["enabled"] = old_enabled
        updated_models[model_name] = dumped

    updates = {
        "models_list": updated_models,
        "updated_at": datetime.now(timezone.utc),
    }

    await config_ref.update(updates)
    print(
        f"  [004_configure_claude_models] ✓ Successfully updated models_list in Firestore "
        f"({len(updated_models)} total models; Claude models: {list(CLAUDE_MODELS.keys())})."
    )

    # Invalidate Redis cache if available
    try:
        from app.core.cache_keys import CacheKeys
        from app.core.redis import redis_cache

        await redis_cache.connect()
        await redis_cache.delete(CacheKeys.system_config())
        await redis_cache.disconnect()
        print("  [004_configure_claude_models] ✓ Invalidated Redis system_config cache")
    except Exception as exc:
        print(f"  [004_configure_claude_models] (Redis cache invalidation skipped: {exc})")


if __name__ == "__main__":
    asyncio.run(migrate())
