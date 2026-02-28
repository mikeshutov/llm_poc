from dataclasses import dataclass

from agent.models.tool_call_trace import ToolCallTrace
from response_layer.models.response_payload import ResponsePayload


@dataclass(frozen=True)
class AgentResult:
    answer: ResponsePayload
    debug_trace: dict
    tool_traces: list[ToolCallTrace]
    goal: str
    goal_reached: bool
    iterations_used: int
