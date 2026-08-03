"""Conversation context management.

Replaces the "send the entire transcript every turn" behaviour with budgeted
retention: estimate tokens, drop the oldest messages while over budget, always
keep the most recent messages. System instructions are injected separately and
are never affected by trimming.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class ContextManager:
    """Trims conversation history before it is sent to the model."""

    TOKENS_PER_CHAR = 4.0  # rough heuristic: ~4 chars per token (English)

    def estimate_text_tokens(self, text: str) -> int:
        if not text:
            return 0
        return max(1, len(text) // int(self.TOKENS_PER_CHAR))

    def estimate_message_tokens(self, message: Any) -> int:
        """Estimate the token cost of a stored message (content + attachments).

        ``message`` is a ``ChatMessageDB``-like object with ``content`` and an
        optional ``attachment_ids`` list.
        """
        tokens = self.estimate_text_tokens(getattr(message, "content", "") or "")
        attachment_ids = getattr(message, "attachment_ids", None) or []
        # Attachments add a fixed per-file cost on top of any text (images,
        # audio, video and PDFs are more token-expensive than their bytes
        # suggest).
        tokens += 1000 * len(attachment_ids)
        return tokens

    def trim(
        self,
        messages: list[Any],
        max_tokens: int,
        keep_recent: int,
        estimator: Callable[[Any], int] | None = None,
    ) -> list[Any]:
        """Drop the oldest messages until the estimated history fits the
        budget, while always retaining at least ``keep_recent`` messages."""
        if max_tokens <= 0:
            return list(messages[-keep_recent:]) if keep_recent > 0 else []

        estimate = estimator or self.estimate_message_tokens
        kept = list(messages)
        total = sum(estimate(m) for m in kept)

        dropped = 0
        while total > max_tokens and len(kept) > max(keep_recent, 0):
            removed = kept.pop(0)
            total -= estimate(removed)
            dropped += 1

        if dropped:
            logger.info(
                "Context trimmed: dropped %d of %d messages (estimated %d tokens → %d)",
                dropped,
                len(messages),
                sum(estimate(m) for m in messages),
                total,
            )
        return kept
