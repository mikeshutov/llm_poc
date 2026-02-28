from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolCallTrace:
    call_index: int
    turn_index: int
    tool_name: str
    status: str
    reason: str | None
    input_payload: dict[str, Any]
    output_payload: dict[str, Any] | None
    error_message: str | None
    duration_ms: int | None
    goal: str | None = None
    done: bool | None = None
