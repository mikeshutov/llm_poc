from __future__ import annotations

from agent_runtime.constants import PLAN_EDGE, SYNTHESIZE_EDGE
from agent_runtime.models.agent_state import AgentState
from agent_runtime.models.evaluation_result import (
    TERMINAL_EVALUATION_STATUSES,
    EVALUATION_STATUS_SATISFIED,
    EVALUATION_STATUS_PARTIAL_BUT_SUFFICIENT,
)


def evaluator_router(state: AgentState) -> str:
    planner_state = state.node_states.planner
    evaluator_state = state.node_states.evaluator
    if state.agent_profile.uses_memory_state and evaluator_state.evaluation_status in {
        EVALUATION_STATUS_SATISFIED, EVALUATION_STATUS_PARTIAL_BUT_SUFFICIENT,
    }:
        if planner_state.plan_count < state.max_turns:
            return PLAN_EDGE
    if planner_state.plan_count >= state.max_turns:
        return SYNTHESIZE_EDGE
    if evaluator_state.evaluation_status in TERMINAL_EVALUATION_STATUSES:
        return SYNTHESIZE_EDGE
    return PLAN_EDGE
