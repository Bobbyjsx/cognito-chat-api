"""Tests for the tool execution loop."""

import pytest

from app.providers.base import (
    ContentPart,
    GenerationEvent,
    GenerationResult,
    ProviderError,
    ToolCall,
)
from app.tools.base import ToolOutput
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry


class EchoTool:
    name = "echo"
    description = "Echoes a message"
    kind = "function"

    @property
    def schema(self):
        return {"type": "object", "properties": {"text": {"type": "string"}}}

    async def execute(self, args):
        return ToolOutput(content={"echoed": args.get("text", "")})


class BoomTool:
    name = "boom"
    description = "Always fails"
    kind = "function"

    @property
    def schema(self):
        return {"type": "object"}

    async def execute(self, args):
        raise RuntimeError("kaboom")


class FakeProvider:
    """Records the contents fed back between rounds and emits scripted events."""

    def __init__(self, script):
        self.script = list(script)
        self.rounds: list[list[ContentPart]] = []

    async def generate_stream(self, model, contents, config=None):
        self.rounds.append(list(contents))
        events = self.script.pop(0) if self.script else [GenerationEvent(type="text", token="done")]
        for event in events:
            yield event

    async def generate(self, model, contents, config=None):
        self.rounds.append(list(contents))
        return self.script.pop(0) if self.script else GenerationResult(text="done", total_tokens=0)


def _echo_call():
    return GenerationEvent(
        type="tool_call",
        tool_call=ToolCall(id="call_1", name="echo", args={"text": "hi"}),
    )


@pytest.mark.asyncio
async def test_streaming_loop_feeds_results_back():
    provider = FakeProvider(
        [
            [_echo_call()],
            [GenerationEvent(type="text", token="final")],
        ]
    )
    registry = ToolRegistry()
    registry.register(EchoTool())
    executor = ToolExecutor(registry, provider)

    events = [
        e
        async for e in executor.generate_stream(
            "gemini-x", [ContentPart(role="user", parts=[{"text": "go"}])]
        )
    ]

    assert [e.type for e in events] == ["tool_call", "tool_result", "text"]

    tool_result = next(e for e in events if e.type == "tool_result")
    assert tool_result.tool_result.output == {"echoed": "hi"}

    # Round 2 contents must contain the function_call + function_response parts
    round_two = provider.rounds[1]
    assert round_two[-2].role == "model"
    assert round_two[-2].parts == [
        {"function_call": {"id": "call_1", "name": "echo", "args": {"text": "hi"}}}
    ]
    assert round_two[-1].role == "user"
    assert round_two[-1].parts == [
        {"function_response": {"id": "call_1", "name": "echo", "response": {"echoed": "hi"}}}
    ]


@pytest.mark.asyncio
async def test_non_streaming_loop_returns_final_result():
    provider = FakeProvider(
        [
            GenerationResult(text="", total_tokens=10, tool_calls=[_echo_call().tool_call]),
            GenerationResult(text="answer", total_tokens=20),
        ]
    )
    registry = ToolRegistry()
    registry.register(EchoTool())
    executor = ToolExecutor(registry, provider)

    result = await executor.generate("gemini-x", [ContentPart(role="user", parts=[{"text": "go"}])])
    assert result.text == "answer"
    assert result.total_tokens == 20


@pytest.mark.asyncio
async def test_unknown_tool_returns_error_result():
    provider = FakeProvider(
        [
            [GenerationEvent(type="tool_call", tool_call=ToolCall(id="call_x", name="missing", args={}))],
            [GenerationEvent(type="text", token="ok")],
        ]
    )
    executor = ToolExecutor(ToolRegistry(), provider)

    events = [e async for e in executor.generate_stream("gemini-x", [])]
    tool_result = next(e for e in events if e.type == "tool_result")
    assert tool_result.tool_result.is_error is True
    assert "Unknown tool" in str(tool_result.tool_result.output["error"])


@pytest.mark.asyncio
async def test_failing_tool_does_not_kill_the_turn():
    provider = FakeProvider(
        [
            [GenerationEvent(type="tool_call", tool_call=ToolCall(id="call_b", name="boom", args={}))],
            [GenerationEvent(type="text", token="recovered")],
        ]
    )
    registry = ToolRegistry()
    registry.register(BoomTool())
    executor = ToolExecutor(registry, provider)

    events = [e async for e in executor.generate_stream("gemini-x", [])]
    tool_result = next(e for e in events if e.type == "tool_result")
    assert tool_result.tool_result.is_error is True
    assert events[-1].type == "text"


@pytest.mark.asyncio
async def test_max_iterations_guard_raises():
    registry = ToolRegistry()
    registry.register(EchoTool())
    always_calls = FakeProvider(
        [
            GenerationResult(text="", total_tokens=0, tool_calls=[_echo_call().tool_call])
            for _ in range(10)
        ]
    )
    executor = ToolExecutor(registry, always_calls, max_iterations=2)

    with pytest.raises(ProviderError, match="Maximum tool call iterations"):
        await executor.generate("gemini-x", [])
