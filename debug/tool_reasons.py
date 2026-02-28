import streamlit as st

from debug.rendering import debug_render_message
from common.message_constants import CONTENT_KEY, ROLE_DEBUG, ROLE_KEY


def build_tool_reason_debug(trace) -> dict:
    return {
        "call_index": trace.call_index,
        "turn_index": trace.turn_index,
        "tool_name": trace.tool_name,
        "status": trace.status,
        "reason": trace.reason,
        "input_payload": trace.input_payload,
        "output_payload": trace.output_payload,
        "goal": trace.goal,
        "done": trace.done,
        "duration_ms": trace.duration_ms,
        "error_message": trace.error_message,
    }


def emit_tool_trace_debug(trace) -> None:
    tool_reason_debug_title = f"**Debug: Tool call result ({trace.tool_name})**"
    tool_reason_debug_payload = build_tool_reason_debug(trace)
    st.session_state.messages.append(
        {
            ROLE_KEY: ROLE_DEBUG,
            CONTENT_KEY: tool_reason_debug_payload,
            "title": tool_reason_debug_title,
        }
    )
    debug_render_message(tool_reason_debug_payload, tool_reason_debug_title)
