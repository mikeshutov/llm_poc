from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict[str, Any]


@dataclass(frozen=True)
class ToolCallResult:
    tool_calls: list[ToolCall]
    tool_calls_by_name: dict[str, list[ToolCall]]  # keeps all calls per name
    raw_message: Any
