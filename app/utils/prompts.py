# Central repository for AI Agent prompts and system instructions


def get_base_system_instructions() -> str:
    """
    Returns the core system instructions for the conversational agent.
    """
    return (
        "You are a helpful, intelligent, and concise AI assistant. "
        "Your primary goal is to provide accurate and actionable answers to the user. "
        "Keep your responses polite and well-structured."
    )
