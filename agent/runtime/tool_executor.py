from typing import Any

from agent.models.tool_call_status import ToolCallStatus
from common.time import now_ms
from tool_registry import call_tool


def execute_tool(name: str, args: dict[str, Any]) -> tuple[str, dict[str, Any] | None, str | None, int]:
    started = now_ms()
    status = ToolCallStatus.SUCCESS.value
    error: str | None = None
    output_payload: dict[str, Any] | None = None
    try:
        output = call_tool(name, args)
        output_payload = output if isinstance(output, dict) else {"value": output}
    except Exception as exc:
        status = ToolCallStatus.ERROR.value
        error = str(exc)
    return status, output_payload, error, now_ms() - started

