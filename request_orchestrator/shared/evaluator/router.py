from __future__ import annotations

from request_orchestrator.constants import PLAN_EDGE, SYNTHESIZE_EDGE
from request_orchestrator.models.agent_state import AgentState
from request_orchestrator.models.evaluation_result import TERMINAL_EVALUATION_STATUSES


def evaluator_router(state: AgentState) -> str:
    planner_state = state.node_states.planner
    evaluator_state = state.node_states.evaluator
    if (
        evaluator_state.evaluation_status in TERMINAL_EVALUATION_STATUSES
        or evaluator_state.goal_reached
        or planner_state.plan_count >= state.max_turns
    ):
        return SYNTHESIZE_EDGE
    return PLAN_EDGE
