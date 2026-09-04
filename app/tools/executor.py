"""Tool execution loop.

Orchestrates the conversation between the model and the application's tools:

1. stream a generation round from the provider
2. detect function-call requests (app-executed tools only — server-side tools
   are handled by the provider itself)
3. execute the requested tools
4. feed the results back into the conversation
5. continue generating until the model finishes or the round limit is hit

Server-side tool events (``code_execution``, ``google_search``) are streamed
through untouched. All events keep the existing SSE compatibility: any tool
surfaces as ``tool_call`` followed by ``tool_result``.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from app.providers.base import (
    BaseProvider,
    ContentPart,
    GenerationConfig,
    GenerationEvent,
    GenerationResult,
    ProviderError,
    ToolResult,
)
from app.tools.base import ToolOutput
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

MAX_TOOL_ITERATIONS = 4


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        provider: BaseProvider | Any | None = None,
        max_iterations: int = MAX_TOOL_ITERATIONS,
    ):
        self.registry = registry
        if provider is None:
            from app.providers.registry import create_default_provider_registry

            provider = create_default_provider_registry()
        self.provider = provider
        self.max_iterations = max_iterations

    def _resolve_provider(self, model: str) -> BaseProvider:
        from app.providers.registry import ProviderRegistry

        if isinstance(self.provider, ProviderRegistry):
            return self.provider.get_for_model(model)
        return self.provider

    # ── streaming path ────────────────────────────────────────────────────────

    async def generate_stream(
        self,
        model: str,
        contents: list[ContentPart],
        config: GenerationConfig | None = None,
    ) -> AsyncIterator[GenerationEvent]:
        """Yields generation events, executing function tools mid-stream."""
        provider = self._resolve_provider(model)
        for _ in range(self.max_iterations):
            function_calls = []
            async for event in provider.generate_stream(model, contents, config):
                yield event
                if event.type == "tool_call" and event.tool_call is not None and event.tool_call.kind == "function":
                    function_calls.append(event.tool_call)

            if not function_calls:
                return

            results = await self.execute_calls(function_calls)
            for result in results:
                yield GenerationEvent(type="tool_result", tool_result=result)

            contents = self._append_tool_round(contents, function_calls, results)

        raise ProviderError("Maximum tool call iterations exceeded.")

    # ── non-streaming path ────────────────────────────────────────────────────

    async def generate(
        self,
        model: str,
        contents: list[ContentPart],
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        """Generate a complete response, executing function tools between
        rounds. Returns the final model result (no tool calls pending)."""
        provider = self._resolve_provider(model)
        for _ in range(self.max_iterations):
            result = await provider.generate(model, contents, config)
            function_calls = [call for call in result.tool_calls if call.kind == "function"]
            if not function_calls:
                return result

            results = await self.execute_calls(function_calls)
            contents = self._append_tool_round(contents, function_calls, results)

        raise ProviderError("Maximum tool call iterations exceeded.")

    # ── tool execution ────────────────────────────────────────────────────────

    async def execute_calls(self, function_calls) -> list[ToolResult]:
        results: list[ToolResult] = []
        for call in function_calls:
            tool = self.registry.function_tool(call.name)
            if tool is None:
                logger.warning("Model requested unknown tool '%s'", call.name)
                results.append(
                    ToolResult(
                        id=call.id,
                        name=call.name,
                        output={"error": f"Unknown tool '{call.name}'"},
                        is_error=True,
                    )
                )
                continue

            try:
                output = await tool.execute(call.args or {})
                if isinstance(output, ToolOutput):
                    content, is_error = output.content, output.is_error
                else:
                    content, is_error = output, False
                results.append(ToolResult(id=call.id, name=call.name, output=content, is_error=is_error))
            except Exception as exc:
                logger.exception("Tool '%s' execution failed", call.name)
                results.append(
                    ToolResult(
                        id=call.id,
                        name=call.name,
                        output={"error": str(exc)},
                        is_error=True,
                    )
                )
        return results

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _append_tool_round(
        contents: list[ContentPart],
        function_calls,
        results: list[ToolResult],
    ) -> list[ContentPart]:
        """Append the model's function-call parts and the app's function
        responses so the provider can continue the conversation."""
        model_parts = []
        for call in function_calls:
            fc_data: dict[str, Any] = {
                "id": call.id,
                "name": call.name,
                "args": call.args or {},
            }
            thought_sig = getattr(call, "thought_signature", None)
            part_data: dict[str, Any] = {"function_call": fc_data}
            if thought_sig is not None:
                part_data["thought_signature"] = thought_sig
                fc_data["thought_signature"] = thought_sig
            model_parts.append(part_data)
        user_parts = [
            {
                "function_response": {
                    "id": result.id,
                    "name": result.name,
                    "response": result.output,
                }
            }
            for result in results
        ]
        return [*contents, ContentPart(role="model", parts=model_parts), ContentPart(role="user", parts=user_parts)]

    # Kept as a module-level constant for external use.
    @property
    def max_rounds(self) -> int:
        return self.max_iterations
