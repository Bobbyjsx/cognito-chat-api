"""Tool registry — the single source of truth for available tools.

The registry is built once at application startup, registers every bundled
tool, and derives the provider tool configuration from the runtime
``allowed_tools`` list. Adding a new tool requires nothing more than a
``BaseTool`` subclass and a ``register()`` call.
"""

from __future__ import annotations

from typing import Any

from app.tools.base import BaseTool
from app.tools.code_execution import CodeExecutionTool
from app.tools.google_search import GoogleSearchTool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool by name. Re-registering a name replaces it."""
        self._tools[tool.name] = tool

    def register_defaults(self) -> None:
        """Register every tool bundled with the application."""
        self.register(CodeExecutionTool())
        self.register(GoogleSearchTool())

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def enabled_names(self, allowed: list[str] | None) -> list[str]:
        """Intersection of the configured tool names and registered tools."""
        if not allowed:
            return []
        return [name for name in allowed if name in self._tools]

    def function_tool(self, name: str) -> BaseTool | None:
        """Look up an app-executed (``kind == "function"``) tool by name."""
        tool = self._tools.get(name)
        return tool if tool is not None and tool.kind == "function" else None

    def to_provider_configs(self, allowed: list[str] | None) -> list[dict[str, Any]]:
        """Convert enabled tools into provider-agnostic tool definitions.

        Each config is a dict with ``kind`` (``"code_execution"``,
        ``"google_search"`` or ``"function"``) plus, for function tools,
        ``name``, ``description`` and ``schema``. Providers translate these
        into native tool objects.
        """
        configs: list[dict[str, Any]] = []
        for name in self.enabled_names(allowed):
            tool = self._tools[name]
            if tool.kind == "server":
                configs.append({"kind": tool.name})
            else:
                configs.append(
                    {
                        "kind": "function",
                        "name": tool.name,
                        "description": tool.description,
                        "schema": tool.schema,
                    }
                )
        return configs
