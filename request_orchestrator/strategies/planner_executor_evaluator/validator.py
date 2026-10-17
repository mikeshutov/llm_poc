from __future__ import annotations

from request_orchestrator.constants import EXECUTE_TOOLS_EDGE, SYNTHESIZE_EDGE
from request_orchestrator.models.agent_state import AgentState


def validator(state: AgentState) -> str:
    planner_state = state.node_states.planner
    if planner_state.tool_calls:
        return EXECUTE_TOOLS_EDGE
    if planner_state.plan is None:
        return SYNTHESIZE_EDGE
    plan = planner_state.plan
    if len(plan.steps) == 0:
        return SYNTHESIZE_EDGE
    # A newly generated executable plan must run before any stale evaluator
    # state is considered.
    return EXECUTE_TOOLS_EDGE
