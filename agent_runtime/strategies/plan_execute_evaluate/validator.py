from __future__ import annotations

from agent_runtime.constants import EXECUTE_TOOLS_EDGE, EVALUATE_EDGE, SYNTHESIZE_EDGE
from agent_runtime.models.agent_state import AgentState


def validator(state: AgentState) -> str:
    planner_state = state.node_states.planner
    if planner_state.plan is None:
        return EVALUATE_EDGE
    plan = planner_state.plan
    if len(plan.steps) == 0:
        return EVALUATE_EDGE
    # A newly generated executable plan must run before any stale evaluator
    # state is considered.
    return EXECUTE_TOOLS_EDGE
