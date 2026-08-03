"""Generic tool framework.

New capabilities are implemented as :class:`BaseTool` subclasses and
registered in the :class:`~app.tools.registry.ToolRegistry`; no changes to the
chat service are required.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from app.providers.base import TOOL_KIND_FUNCTION, TOOL_KIND_SERVER


@dataclass
class ToolOutput:
    """Result of executing a tool."""

    content: Any = field(default_factory=dict)
    is_error: bool = False


class BaseTool(ABC):
    """Interface every tool implements.

    ``kind`` distinguishes app-executed tools (``function``) from tools the
    AI provider executes server-side (``server``, e.g. Gemini's built-in
    code execution / Google Search grounding).
    """

    kind: str = TOOL_KIND_FUNCTION
    name: str = ""
    description: str = ""

    @property
    @abstractmethod
    def schema(self) -> dict[str, Any] | None:
        """JSON Schema for the tool's parameters (``None`` for server tools)."""

    async def execute(self, args: dict[str, Any]) -> ToolOutput:
        """Execute the tool with the given arguments.

        Only relevant for ``kind == "function"`` tools. Server-side tools are
        executed by the provider and override ``execute`` only to document
        that.
        """
        raise NotImplementedError(
            f"Tool '{self.name}' is a server-side tool and cannot be executed by the application."
        )


class ServerSideTool(BaseTool):
    """Base class for tools executed by the AI provider (e.g. Gemini)."""

    kind: str = TOOL_KIND_SERVER

    @property
    def schema(self) -> dict[str, Any] | None:
        return None
