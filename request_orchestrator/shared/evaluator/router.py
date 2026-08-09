from __future__ import annotations

from request_orchestrator.constants import PLAN_EDGE, SYNTHESIZE_EDGE
from request_orchestrator.models.agent_state import AgentState


def evaluator_router(state: AgentState) -> str:
    if state.goal_reached or len(state.iteration_trace) >= state.max_turns:
        return SYNTHESIZE_EDGE
    return PLAN_EDGE
