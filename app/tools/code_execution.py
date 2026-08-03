"""Gemini's built-in code execution tool.

Execution happens server-side on Google's infrastructure: the model emits
``executable_code`` parts and receives ``code_execution_result`` parts without
any application involvement. The provider surfaces both as
``tool_call``/``tool_result`` streaming events for the frontend.
"""

from __future__ import annotations

from app.tools.base import ServerSideTool


class CodeExecutionTool(ServerSideTool):
    name = "code_execution"
    description = (
        "Executes Python code in a sandboxed runtime on the model provider's "
        "servers. Use for calculations, data analysis, or any task that "
        "benefits from running code."
    )
