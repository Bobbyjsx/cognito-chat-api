from __future__ import annotations

import logging
from typing import Any

from app.providers.base import TOOL_KIND_SERVER
from app.tools.base import BaseTool, ToolOutput

logger = logging.getLogger(__name__)


class GoogleSearchTool(BaseTool):
    """Google Search tool.

    For Gemini, this maps to server-side Google Search grounding.
    For non-server providers (e.g. Claude via Bedrock), this provides a function
    tool execution path that searches the live web and returns grounded sources
    and synthesis.
    """

    name = "google_search"
    kind = TOOL_KIND_SERVER
    description = (
        "Searches the live web and grounds the answer in current, "
        "verifiable sources. Use for questions about recent events, facts, current dates, "
        "or anything requiring up-to-date information."
    )

    @property
    def schema(self) -> dict[str, Any] | None:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up on the live web.",
                },
            },
            "required": ["query"],
        }

    async def execute(self, args: dict[str, Any]) -> ToolOutput:
        """Execute web search and return verifiable sources and summary."""
        query = ""
        if isinstance(args, dict):
            raw_query = args.get("query")
            if raw_query and isinstance(raw_query, str):
                query = raw_query.strip()
            elif "queries" in args:
                raw_queries = args.get("queries")
                if isinstance(raw_queries, list):
                    query = " ".join(str(q).strip() for q in raw_queries if str(q).strip())
                elif isinstance(raw_queries, str):
                    query = raw_queries.strip()
        if not query:
            return ToolOutput(content={"error": "Empty search query.", "query": ""}, is_error=True)

        try:
            import os

            from google import genai
            from google.genai import types

            from app.core.config import settings

            api_key = settings.gemini_api_key or os.getenv("GEMINI_API_KEY")
            client = genai.Client(api_key=api_key) if api_key else genai.Client()

            prompt = f"Search Google and provide the latest facts, data, and sources for: {query}"
            response = await client.aio.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                    temperature=0.0,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                ),
            )

            sources: list[dict[str, str]] = []
            grounding = None
            if response.candidates:
                grounding = getattr(response.candidates[0], "grounding_metadata", None)
            if grounding is None:
                grounding = getattr(response, "grounding_metadata", None)

            if grounding:
                for gchunk in getattr(grounding, "grounding_chunks", None) or []:
                    web = getattr(gchunk, "web", None)
                    if web:
                        sources.append(
                            {
                                "title": getattr(web, "title", None) or "Web Source",
                                "uri": getattr(web, "uri", None) or "",
                            }
                        )

            summary = response.text or ""
            return ToolOutput(
                content={
                    "query": query,
                    "sources": sources,
                    "summary": summary,
                }
            )
        except Exception as exc:
            logger.exception("Google search tool execution failed")
            return ToolOutput(content={"error": f"Search failed: {exc}", "query": query}, is_error=True)
