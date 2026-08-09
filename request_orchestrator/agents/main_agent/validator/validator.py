from __future__ import annotations

from request_orchestrator.constants import EXECUTE_TOOLS_EDGE, SYNTHESIZE_EDGE
from request_orchestrator.models.agent_state import AgentState


def validator(state: AgentState) -> str:
    if state.goal_reached or not state.iteration_trace:
        return SYNTHESIZE_EDGE

    last_iteration = state.iteration_trace[-1]
    plan = last_iteration.plan
    if plan is None or plan.final_answer or len(plan.steps) == 0:
        return SYNTHESIZE_EDGE

    return EXECUTE_TOOLS_EDGE
