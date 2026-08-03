# Central repository for AI Agent prompts and system instructions


def get_base_system_instructions() -> str:
    """Standard system instructions for Gemini chat sessions."""
    return (
        "You are Cognito, an advanced AI assistant created to be helpful, concise, and clear. "
        "Format responses cleanly with Markdown when applicable."
    )
