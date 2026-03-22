from __future__ import annotations

from agent.agentstate.model import AgentState
from agent.graph_constants import PLAN_EDGE, SYNTHESIZE_EDGE


def router(state: AgentState) -> str:
    if state.goal_reached:
        return SYNTHESIZE_EDGE

    if len(state.iteration_trace) >= state.max_turns:
        return SYNTHESIZE_EDGE

    last_iteration = state.iteration_trace[-1] if state.iteration_trace else None
    if last_iteration and last_iteration.plan and not last_iteration.plan.needs_replan:
        return SYNTHESIZE_EDGE

    return PLAN_EDGE
