from __future__ import annotations

from request_orchestrator.models.agent_state import AgentState
from request_orchestrator.constants import EVALUATE_EDGE, PLAN_EDGE, SYNTHESIZE_EDGE


def router(state: AgentState) -> str:
    if state.goal_reached:
        return SYNTHESIZE_EDGE

    if len(state.iteration_trace) >= state.max_turns:
        return SYNTHESIZE_EDGE

    last_iteration = state.iteration_trace[-1] if state.iteration_trace else None
    if last_iteration and last_iteration.results:
        return EVALUATE_EDGE

    return PLAN_EDGE
