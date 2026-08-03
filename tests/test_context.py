"""Tests for context management (history trimming)."""

from app.models.chats import ChatMessageDB
from app.services.context import ContextManager


def _msg(content: str, attachment_ids=None):
    return ChatMessageDB(
        session_id="00000000-0000-0000-0000-000000000000",
        role="user",
        content=content,
        attachment_ids=attachment_ids or [],
    )


def test_trim_keeps_most_recent_when_within_budget():
    manager = ContextManager()
    messages = [_msg("a" * 100), _msg("b" * 100), _msg("c" * 100)]
    kept = manager.trim(messages, max_tokens=100_000, keep_recent=2)
    assert len(kept) == 3


def test_trim_drops_oldest_when_over_budget():
    manager = ContextManager()
    messages = [_msg("x" * 1000) for _ in range(10)]  # ~250 tokens each = 2500
    kept = manager.trim(messages, max_tokens=500, keep_recent=2)
    assert len(kept) >= 2
    # Never drops the most recent message
    assert kept[-1].content == messages[-1].content


def test_trim_always_keeps_at_least_keep_recent():
    manager = ContextManager()
    messages = [_msg("y" * 10_000) for _ in range(5)]
    kept = manager.trim(messages, max_tokens=10, keep_recent=3)
    assert len(kept) == 3
    assert kept[-1].content == messages[-1].content


def test_trim_respects_custom_estimator():
    manager = ContextManager()
    messages = [_msg("z"), _msg("z")]
    # Every message costs exactly 100 tokens via the custom estimator
    kept = manager.trim(messages, max_tokens=150, keep_recent=1, estimator=lambda m: 100)
    assert len(kept) == 1


def test_trim_zero_budget_keeps_recent_only():
    manager = ContextManager()
    messages = [_msg("a"), _msg("b"), _msg("c")]
    kept = manager.trim(messages, max_tokens=0, keep_recent=2)
    assert [m.content for m in kept] == ["b", "c"]


def test_attachments_add_fixed_token_cost():
    manager = ContextManager()
    messages = [_msg("short text", attachment_ids=["att-1"])]
    assert manager.estimate_message_tokens(messages[0]) > manager.estimate_text_tokens("short text")
