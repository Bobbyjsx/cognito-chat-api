"""Unit tests for Telemetry sinks and event emission."""

import pytest

from app.ai.router.schemas import RoutingTelemetry, TaskType
from app.ai.router.telemetry import (
    CompositeTelemetrySink,
    InMemoryTelemetrySink,
    LoggingTelemetrySink,
)
from app.models.config import ModelProvider, RoutingMode


@pytest.mark.asyncio
async def test_in_memory_telemetry_sink():
    sink = InMemoryTelemetrySink(maxlen=5)
    for i in range(10):
        event = RoutingTelemetry(
            selected_model=f"model-{i}",
            provider=ModelProvider.GOOGLE,
            routing_mode=RoutingMode.BALANCED,
            task_type=TaskType.CODING,
            complexity=0.5,
            score=0.8,
        )
        await sink.emit(event)

    assert len(sink) == 5
    recent = sink.get_recent(3)
    assert len(recent) == 3
    assert recent[-1].selected_model == "model-9"


@pytest.mark.asyncio
async def test_composite_telemetry_sink():
    in_memory = InMemoryTelemetrySink()
    logging_sink = LoggingTelemetrySink()
    composite = CompositeTelemetrySink([in_memory, logging_sink])

    event = RoutingTelemetry(
        selected_model="gemini-3.6-flash",
        provider=ModelProvider.GOOGLE,
        routing_mode=RoutingMode.BALANCED,
        task_type=TaskType.CODING,
        complexity=0.8,
        score=0.9,
    )
    await composite.emit(event)
    assert len(in_memory) == 1
