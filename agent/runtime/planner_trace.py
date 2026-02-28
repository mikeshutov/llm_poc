from typing import Any

from agent.models.tool_call_trace import ToolCallTrace


def build_tool_trace(
    *,
    call_index: int,
    turn_index: int,
    tool_name: str,
    status: str,
    reason: str,
    input_payload: dict[str, Any],
    output_payload: dict[str, Any] | None,
    error_message: str | None,
    duration_ms: int,
    goal: str,
    done: bool | None = None,
) -> ToolCallTrace:
    return ToolCallTrace(
        call_index=call_index,
        turn_index=turn_index,
        tool_name=tool_name,
        status=status,
        reason=reason,
        input_payload=input_payload,
        output_payload=output_payload,
        error_message=error_message,
        duration_ms=duration_ms,
        goal=goal,
        done=done,
    )

