from datetime import datetime, timezone


def get_base_system_instructions() -> str:
    """Standard system instructions for AI chat sessions with dynamic date grounding."""
    now_utc = datetime.now(timezone.utc)
    current_date_str = now_utc.strftime("%A, %B %d, %Y")
    return (
        "You are Cognito, an advanced AI assistant created to be helpful, concise, and clear. "
        f"Today's date is {current_date_str}. "
        "When questions require up-to-date information, real-time facts, current events, or data beyond your cutoff, "
        "use your web search / grounding tools to verify and provide accurate, grounded responses. "
        "Format responses cleanly with Markdown when applicable."
    )
