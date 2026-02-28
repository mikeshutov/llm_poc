from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any


class ToolRegistryError(Exception):
    pass


class UnknownToolError(ToolRegistryError):
    pass


class InvalidToolParamsError(ToolRegistryError):
    pass


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, fn: Callable[..., Any]) -> None:
        if not name or not isinstance(name, str):
            raise ToolRegistryError("Tool name must be a non-empty string.")
        if not callable(fn):
            raise ToolRegistryError(f"Tool '{name}' must be callable.")
        self._tools[name] = fn

    def get(self, name: str) -> Callable[..., Any]:
        tool = self._tools.get(name)
        if tool is None:
            raise UnknownToolError(f"Unknown tool '{name}'.")
        return tool

    def has(self, name: str) -> bool:
        return name in self._tools

    def list_tools(self) -> list[str]:
        return sorted(self._tools.keys())

    def call_tool(self, name: str, params: dict[str, Any] | None = None) -> Any:
        call_params = params or {}
        if not isinstance(call_params, dict):
            raise InvalidToolParamsError("Tool params must be a dictionary when provided.")

        fn = self.get(name)

        try:
            inspect.signature(fn).bind(**call_params)
        except TypeError as exc:
            raise InvalidToolParamsError(f"Invalid params for tool '{name}': {exc}") from exc

        return fn(**call_params)
