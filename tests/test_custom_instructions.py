from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from app.models.users import UserDB
from app.router.auth import CustomInstructionsRequest, update_custom_instructions
from app.services.quota import QuotaService
from app.utils.prompts import get_base_system_instructions, get_user_system_instructions


def test_get_user_system_instructions_without_custom():
    instructions = get_user_system_instructions()
    assert instructions == get_base_system_instructions()
    assert "User Custom Instructions" not in instructions


def test_get_user_system_instructions_with_custom():
    custom = "Always respond in bullet points and format code using Python type annotations."
    instructions = get_user_system_instructions(custom)
    assert get_base_system_instructions() in instructions
    assert "# User Custom Instructions" in instructions
    assert custom in instructions


def test_user_response_includes_subscription_status_and_custom_instructions():
    user = UserDB(
        id="test-user-id",
        email="test@example.com",
        hashed_password="hash",
        custom_instructions="Be very brief.",
    )
    res = QuotaService.build_user_response(
        user,
        subscription_tier="premium",
        subscription_status="active",
        is_subscribed=True,
    )
    assert res.tier == "premium"
    assert res.subscription_status == "active"
    assert res.is_subscribed is True
    assert res.custom_instructions == "Be very brief."

    data = res.model_dump(mode="json")
    assert data["subscription_status"] == "active"
    assert data["is_subscribed"] is True
    assert data["custom_instructions"] == "Be very brief."


@pytest.mark.asyncio
async def test_update_custom_instructions_forbidden_for_free_user():
    current_user = UserDB(id="user-1", email="test@test.com", hashed_password="pw")
    db_mock = MagicMock()

    with patch("app.router.auth.lookup_active_tier", new_callable=AsyncMock) as mock_tier:
        mock_tier.return_value = "free"
        req = CustomInstructionsRequest(custom_instructions="Hello instructions")
        with pytest.raises(HTTPException) as exc_info:
            await update_custom_instructions(
                request=req,
                current_user=current_user,
                db=db_mock,
            )
        assert exc_info.value.status_code == 403
        assert "Premium" in exc_info.value.detail


@pytest.mark.asyncio
async def test_update_custom_instructions_allowed_for_premium_user():
    current_user = UserDB(id="user-2", email="prem@test.com", hashed_password="pw")
    db_mock = MagicMock()

    with (
        patch("app.router.auth.lookup_active_tier", new_callable=AsyncMock) as mock_tier,
        patch("app.router.auth.UserRepository") as mock_user_repo_cls,
    ):
        mock_tier.return_value = "premium"
        mock_repo = MagicMock()
        mock_repo.update_custom_instructions = AsyncMock()
        mock_user_repo_cls.return_value = mock_repo

        req = CustomInstructionsRequest(custom_instructions="Be very concise and structured.")
        res = await update_custom_instructions(
            request=req,
            current_user=current_user,
            db=db_mock,
        )
        assert res["custom_instructions"] == "Be very concise and structured."
        mock_repo.update_custom_instructions.assert_awaited_once_with(
            current_user.id, "Be very concise and structured."
        )


def test_custom_instructions_request_character_limit():
    from pydantic import ValidationError

    # Under limit (1500 chars) is valid
    valid_req = CustomInstructionsRequest(custom_instructions="a" * 1500)
    assert len(valid_req.custom_instructions) == 1500

    # Over limit (1501 chars) raises validation error
    with pytest.raises(ValidationError):
        CustomInstructionsRequest(custom_instructions="a" * 1501)


def test_prompt_tag_extraction_and_llm_expansion():
    import base64

    from app.utils.prompts import expand_message_for_llm, extract_prompt_tags, get_clean_message_text

    # 1. Built-in prompt resolution by ID
    msg = "[prompt:eng-code-review|Code Review & Security Audit] Please check my code."
    clean, prompts = extract_prompt_tags(msg)
    assert clean == "Please check my code."
    assert len(prompts) == 1
    assert prompts[0]["id"] == "eng-code-review"
    assert prompts[0]["title"] == "Code Review & Security Audit"
    assert "Perform a rigorous code review" in prompts[0]["prompt"]

    expanded = expand_message_for_llm(msg)
    assert "Perform a rigorous code review" in expanded
    assert "Please check my code." in expanded
    assert expanded.startswith("[this skill/prompt - eng-code-review] Please check my code.")
    assert "Prompt details:\n- eng-code-review - Perform a rigorous code review" in expanded
    assert "[prompt:" not in expanded

    # 2. Custom prompt resolution via base64
    custom_template = "You are an expert Chef. Critique this recipe:"
    b64 = base64.b64encode(custom_template.encode("utf-8")).decode("utf-8")
    custom_msg = f"[prompt:custom-123|Chef Review|{b64}] Spaghetti carbonara with cream."

    clean_custom, custom_prompts = extract_prompt_tags(custom_msg)
    assert clean_custom == "Spaghetti carbonara with cream."
    assert custom_prompts[0]["title"] == "Chef Review"
    assert custom_prompts[0]["prompt"] == custom_template

    expanded_custom = expand_message_for_llm(custom_msg)
    assert custom_template in expanded_custom
    assert "Spaghetti carbonara with cream." in expanded_custom
    assert "[prompt:" not in expanded_custom

    # 3. Multiple inline prompts preserve task-specific placement and append definitions once.
    second_template = "Turn the result into a concise executive summary."
    second_b64 = base64.b64encode(second_template.encode("utf-8")).decode("utf-8")
    multi = (
        f"Use [prompt:custom-123|Chef Review|{b64}] on the recipe, then "
        f"[prompt:summary|Executive Summary|{second_b64}] for the final answer."
    )
    expanded_multi = expand_message_for_llm(multi)
    assert expanded_multi.startswith(
        "Use [this skill/prompt - custom-123] on the recipe, then [this skill/prompt - summary] for the final answer."
    )
    assert f"- custom-123 - {custom_template}" in expanded_multi
    assert f"- summary - {second_template}" in expanded_multi

    # 4. Clean message text helper
    assert get_clean_message_text(msg) == "Please check my code."
    assert get_clean_message_text("[prompt:eng-refactor|Refactor]") == "Refactor"
