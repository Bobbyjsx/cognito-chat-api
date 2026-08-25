"""Smart Model Router Exceptions."""

from __future__ import annotations


class RoutingError(Exception):
    """Base exception for all smart model router errors."""


class NoEligibleModelsError(RoutingError):
    """Raised when no supported models satisfy the request's hard constraints."""

    def __init__(
        self,
        message: str = "No eligible models found for the given request constraints.",
        filtered_reasons: dict[str, str] | None = None,
    ):
        super().__init__(message)
        self.filtered_reasons = filtered_reasons or {}


class AnalyzerError(RoutingError):
    """Raised when request analysis fails."""


class InvalidPolicyError(RoutingError):
    """Raised when a routing policy configuration is invalid."""
