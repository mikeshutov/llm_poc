from __future__ import annotations

import ast
from typing import Any

from agent.models.tool_call_status import ToolCallStatus
from agent.models.tool_call_trace import ToolCallTrace
from common import now_ms
from debug import emit_tool_trace_debug
from langchainagent.tools import LANGCHAIN_TOOLS


class ToolRegistryError(Exception):
    pass


class UnknownToolError(ToolRegistryError):
    pass


class InvalidToolParamsError(ToolRegistryError):
    pass


class LangChainToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Any] = {}
        self._call_index = 0

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

    def list_tools(self) -> list[str]:
        return sorted(self._tools.keys())

    def call_tool(self, name: str, tool_input: Any = None) -> Any:
        tool = self.get(name)
        normalized_input = _normalize_tool_input(tool_input, tool)
        started_at = now_ms()
        call_index = self._call_index
        self._call_index += 1

        try:
            result = tool.invoke(normalized_input)
        except Exception as exc:
            _emit_debug_trace(
                ToolCallTrace(
                    call_index=call_index,
                    turn_index=0,
                    tool_name=name,
                    status=ToolCallStatus.ERROR.value,
                    reason="langchain tool invocation",
                    input_payload=_debug_payload(normalized_input),
                    output_payload=None,
                    error_message=str(exc),
                    duration_ms=now_ms() - started_at,
                )
            )
            raise

        _emit_debug_trace(
            ToolCallTrace(
                call_index=call_index,
                turn_index=0,
                tool_name=name,
                status=ToolCallStatus.SUCCESS.value,
                reason="langchain tool invocation",
                input_payload=_debug_payload(normalized_input),
                output_payload=_debug_payload(result),
                error_message=None,
                duration_ms=now_ms() - started_at,
            )
        )
        return result


def _normalize_tool_input(tool_input: Any, tool: Any) -> Any:
    if tool_input is None:
        return {}
    if isinstance(tool_input, str):
        stripped = tool_input.strip()
        if not stripped:
            return {}
        try:
            parsed = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            parsed_kwargs = _parse_keyword_args(stripped)
            if parsed_kwargs is not None:
                return parsed_kwargs
            return stripped
        return _coerce_sequence_input(parsed, tool)
    return _coerce_sequence_input(tool_input, tool)


def _parse_keyword_args(raw_input: str) -> dict[str, Any] | None:
    try:
        parsed = ast.parse(f"_tool({raw_input})", mode="eval")
    except SyntaxError:
        return None

    call = parsed.body
    if not isinstance(call, ast.Call) or call.args:
        return None

    params: dict[str, Any] = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            return None
        try:
            params[keyword.arg] = ast.literal_eval(keyword.value)
        except (SyntaxError, ValueError):
            return None
    return params


def _coerce_sequence_input(tool_input: Any, tool: Any) -> Any:
    if not isinstance(tool_input, (list, tuple)):
        return tool_input

    args_schema = getattr(tool, "args_schema", None)
    model_fields = getattr(args_schema, "model_fields", None)
    if not isinstance(model_fields, dict) or not model_fields:
        return tool_input

    field_names = list(model_fields.keys())
    if len(tool_input) == 1:
        return {field_names[0]: tool_input[0]}

    positional: dict[str, Any] = {}
    extras: list[Any] = []

    for value, field_name in zip(tool_input, field_names):
        field_info = model_fields[field_name]
        field_annotation = getattr(field_info, "annotation", None)
        if field_name != field_names[0] and _expects_mapping(field_annotation) and not isinstance(value, dict):
            extras.append(value)
            continue
        positional[field_name] = value

    if len(tool_input) > len(field_names):
        extras.extend(tool_input[len(field_names):])

    first_field = field_names[0]
    if extras and _expects_string(model_fields[first_field]):
        first_value = positional.get(first_field)
        string_parts = [str(v) for v in ([first_value] if first_value is not None else []) + extras]
        positional[first_field] = " ".join(part for part in string_parts if part).strip()
        extras = []

    return positional if positional else tool_input


def _expects_mapping(annotation: Any) -> bool:
    annotation_text = str(annotation)
    return "dict" in annotation_text.lower() or "Args" in annotation_text


def _expects_string(field_info: Any) -> bool:
    annotation_text = str(getattr(field_info, "annotation", ""))
    return "str" in annotation_text.lower()


def _debug_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {"value": value}


def _emit_debug_trace(trace: ToolCallTrace) -> None:
    try:
        emit_tool_trace_debug(trace)
    except Exception:
        pass


GLOBAL_TOOL_REGISTRY = LangChainToolRegistry()
_REGISTERED_DEFAULTS = False


def register_default_tools() -> None:
    global _REGISTERED_DEFAULTS
    if _REGISTERED_DEFAULTS:
        return

    for tool in LANGCHAIN_TOOLS:
        GLOBAL_TOOL_REGISTRY.register(tool)

    _REGISTERED_DEFAULTS = True


def call_tool(name: str, tool_input: Any = None) -> Any:
    register_default_tools()
    return GLOBAL_TOOL_REGISTRY.call_tool(name=name, tool_input=tool_input)


__all__ = [
    "GLOBAL_TOOL_REGISTRY",
    "ToolRegistryError",
    "UnknownToolError",
    "InvalidToolParamsError",
    "register_default_tools",
    "call_tool",
]
