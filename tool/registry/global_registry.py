from __future__ import annotations

from typing import Any

from rendering.debug import emit_debug_message
from tool.tools import tools


class ToolRegistryError(Exception):
    pass


class UnknownToolError(ToolRegistryError):
    pass


class DisallowedToolError(ToolRegistryError):
    pass


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Any] = {}

    def register(self, tool: Any) -> None:
        name = getattr(tool, "name", None)
        invoke = getattr(tool, "invoke", None)
        if not isinstance(name, str) or not name:
            raise ToolRegistryError("Tool must define a non-empty string name.")
        if not callable(invoke):
            raise ToolRegistryError(f"Tool '{name}' must expose an invoke method.")
        self._tools[name] = tool

    def get(self, name: str) -> Any:
        tool = self._tools.get(name)
        if tool is None:
            raise UnknownToolError(f"Unknown tool '{name}'.")
        return tool

    def call_tool(self, name: str, tool_input: Any = None, *, allowed_tool_names: set[str] | None = None) -> Any:
        if allowed_tool_names is not None and name not in allowed_tool_names:
            raise DisallowedToolError(f"Tool '{name}' is not allowed for this agent.")
        tool = self.get(name)
        try:
            return tool.invoke(tool_input or {})
        except Exception as exc:
            emit_debug_message(
                content={
                    "tool": name,
                    "tool_input": tool_input,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                content_title="Exception Occurred",
            )
            raise


GLOBAL_TOOL_REGISTRY = ToolRegistry()
_REGISTERED_DEFAULTS = False


def register_default_tools() -> None:
    global _REGISTERED_DEFAULTS
    if _REGISTERED_DEFAULTS:
        return
    for tool in tools:
        GLOBAL_TOOL_REGISTRY.register(tool)
    _REGISTERED_DEFAULTS = True


def call_tool(name: str, tool_input: Any = None, *, allowed_tool_names: set[str] | None = None) -> Any:
    register_default_tools()
    return GLOBAL_TOOL_REGISTRY.call_tool(name=name, tool_input=tool_input, allowed_tool_names=allowed_tool_names)


__all__ = [
    "DisallowedToolError",
    "GLOBAL_TOOL_REGISTRY",
    "ToolRegistryError",
    "UnknownToolError",
    "register_default_tools",
    "call_tool",
]
