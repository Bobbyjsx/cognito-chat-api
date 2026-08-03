"""Google Search tool.

Implemented as Gemini's server-side "Grounding with Google Search": when the
tool is enabled in runtime configuration, the model can ground its answers in
live web results. Search happens on the provider side; sources surface as
``tool_call``/``tool_result`` streaming events via grounding metadata.

Isolated in its own module so a future provider can replace it without
touching the registry or the chat service.
"""

from __future__ import annotations

from app.tools.base import ServerSideTool


class GoogleSearchTool(ServerSideTool):
    name = "google_search"
    description = (
        "Searches the live web and grounds the model's answer in current, "
        "verifiable sources. Use for questions about recent events, facts, or "
        "anything requiring up-to-date information."
    )
