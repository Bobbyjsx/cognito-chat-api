"""Telemetry emitters and interfaces for tracking model routing decisions."""

from __future__ import annotations

import collections
import logging
from abc import ABC, abstractmethod

from app.ai.router.schemas import RoutingTelemetry

logger = logging.getLogger("app.ai.router.telemetry")


class TelemetrySink(ABC):
    """Abstract interface for routing telemetry sinks."""

    @abstractmethod
    async def emit(self, event: RoutingTelemetry) -> None:
        """Record or publish a routing telemetry event."""


class LoggingTelemetrySink(TelemetrySink):
    """Logs structured routing telemetry without exposing sensitive prompt contents."""

    async def emit(self, event: RoutingTelemetry) -> None:
        provider_str = event.provider.value if hasattr(event.provider, "value") else str(event.provider)
        mode_str = event.routing_mode.value if hasattr(event.routing_mode, "value") else str(event.routing_mode)
        task_str = event.task_type.value if hasattr(event.task_type, "value") else str(event.task_type)
        logger.info(
            "[SmartRouter][Telemetry] Decision: id=%s model=%s provider=%s mode=%s task=%s complexity=%.2f score=%.4f "
            "candidates=%d filtered=%d latency=%.2fms is_fallback=%s",
            event.request_id,
            event.selected_model,
            provider_str,
            mode_str,
            task_str,
            event.complexity,
            event.score,
            len(event.candidate_models),
            len(event.filtered_out_models),
            event.total_latency_ms or 0.0,
            event.is_fallback,
            extra={"telemetry": event.model_dump(mode="json")},
        )


class InMemoryTelemetrySink(TelemetrySink):
    """Stores the latest N telemetry events in memory for testing, monitoring, and debugging."""

    def __init__(self, maxlen: int = 1000):
        self._events: collections.deque[RoutingTelemetry] = collections.deque(maxlen=maxlen)

    async def emit(self, event: RoutingTelemetry) -> None:
        self._events.append(event)

    def get_recent(self, limit: int = 50) -> list[RoutingTelemetry]:
        """Return the most recent telemetry events."""
        events = list(self._events)
        return events[-limit:]

    def clear(self) -> None:
        self._events.clear()

    def __len__(self) -> int:
        return len(self._events)


class CompositeTelemetrySink(TelemetrySink):
    """Broadcasts telemetry events to multiple sinks concurrently."""

    def __init__(self, sinks: list[TelemetrySink] | None = None):
        self.sinks = sinks or []

    def add_sink(self, sink: TelemetrySink) -> None:
        self.sinks.append(sink)

    async def emit(self, event: RoutingTelemetry) -> None:
        for sink in self.sinks:
            try:
                await sink.emit(event)
            except Exception as exc:
                logger.warning("Failed to emit telemetry event to %s: %s", type(sink).__name__, exc)
