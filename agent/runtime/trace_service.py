from __future__ import annotations

from typing import Any

from agent.runtime.runtime_utils import append_trace
from agent.state import _AgentState


def trace_and_advance(
    state: _AgentState,
    *,
    turn_index: int,
    tool_name: str,
    status: str,
    reason: str,
    input_payload: dict[str, Any],
    output_payload: dict[str, Any] | None = None,
    error_message: str | None = None,
    goal: str | None = None,
    done: bool | None = None,
    duration_ms: int | None = None,
) -> int:
    next_call_index = append_trace(
        state.tool_traces,
        call_index=state.loop_state.call_index,
        turn_index=turn_index,
        tool_name=tool_name,
        status=status,
        reason=reason,
        input_payload=input_payload,
        output_payload=output_payload,
        error_message=error_message,
        goal=goal,
        done=done,
        duration_ms=duration_ms,
    )
    state.loop_state.call_index = next_call_index
    return next_call_index
