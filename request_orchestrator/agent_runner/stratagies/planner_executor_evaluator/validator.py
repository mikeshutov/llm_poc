from __future__ import annotations

from request_orchestrator.constants import EXECUTE_TOOLS_EDGE, SYNTHESIZE_EDGE
from request_orchestrator.models.agent_state import AgentState


def validator(state: AgentState) -> str:
    planner_state = state.node_states.planner
    evaluator_state = state.node_states.evaluator
    if evaluator_state.goal_reached or planner_state.plan is None:
        return SYNTHESIZE_EDGE

    plan = planner_state.plan
    if plan is None or len(plan.steps) == 0:
        return SYNTHESIZE_EDGE

    return EXECUTE_TOOLS_EDGE
