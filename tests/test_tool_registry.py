"""Tests for the tool framework and registry."""

from app.tools.base import BaseTool, ToolOutput
from app.tools.registry import ToolRegistry


class AddTool(BaseTool):
    name = "add"
    description = "Adds two numbers"

    @property
    def schema(self):
        return {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}}

    async def execute(self, args):
        return ToolOutput(content=args.get("a", 0) + args.get("b", 0))


def test_default_registry_contains_bundled_tools():
    registry = ToolRegistry()
    registry.register_defaults()
    assert set(registry.names()) == {"code_execution", "google_search"}


def test_register_custom_tool_and_lookup():
    registry = ToolRegistry()
    registry.register(AddTool())
    tool = registry.get("add")
    assert tool is not None
    assert tool.description.startswith("Adds")


def test_enabled_names_respects_config():
    registry = ToolRegistry()
    registry.register_defaults()
    assert registry.enabled_names(["google_search"]) == ["google_search"]
    assert registry.enabled_names(["google_search", "code_execution", "nonexistent"]) == [
        "google_search",
        "code_execution",
    ]
    assert registry.enabled_names(None) == []


def test_to_provider_configs_maps_kinds():
    registry = ToolRegistry()
    registry.register_defaults()
    registry.register(AddTool())

    configs = registry.to_provider_configs(["code_execution", "google_search", "add"])
    by_kind = {c["kind"]: c for c in configs}
    assert by_kind["code_execution"] == {"kind": "code_execution"}
    assert by_kind["google_search"] == {"kind": "google_search"}
    assert by_kind["function"]["name"] == "add"
    assert by_kind["function"]["description"] == "Adds two numbers"
    assert by_kind["function"]["schema"]["properties"]["a"]["type"] == "number"


def test_to_provider_configs_ignores_unknown_tools():
    registry = ToolRegistry()
    registry.register_defaults()
    assert registry.to_provider_configs(["nonexistent", "code_execution"]) == [{"kind": "code_execution"}]


def test_function_tool_only_returns_app_executed_tools():
    registry = ToolRegistry()
    registry.register_defaults()
    assert registry.function_tool("code_execution") is None  # server-side
    registry.register(AddTool())
    assert registry.function_tool("add") is not None
