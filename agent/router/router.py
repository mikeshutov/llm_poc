from __future__ import annotations

from agent.agentstate.model import AgentState
from agent.graph_constants import PLAN_EDGE, SYNTHESIZE_EDGE


def router(state: AgentState) -> str:
    # Stop if goal reached
    if state.goal_reached:
        return SYNTHESIZE_EDGE
    # Stop if max iterations reached
    if len(state.iteration_trace) >= state.max_turns:
        return SYNTHESIZE_EDGE
    # Otherwise continue planning
    return PLAN_EDGE
