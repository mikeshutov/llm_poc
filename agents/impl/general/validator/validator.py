from __future__ import annotations

from agents.agentstate.model import AgentState
from agents.graph_constants import EXECUTE_TOOLS_EDGE, SYNTHESIZE_EDGE


def validator(state: AgentState) -> str:
    # Potentially implement this as a fallback i.e a repair node which essentially calls the planner with more context
    #    if plan is None:
    #        return "repair"

    # A few cases where we want to go to the solve step
    if state.goal_reached or not state.iteration_trace:
        return SYNTHESIZE_EDGE

    last_iteration = state.iteration_trace[-1]
    if last_iteration.plan is None or last_iteration.plan.final_answer or len(last_iteration.plan.steps) == 0:
        return SYNTHESIZE_EDGE
    
    
    return EXECUTE_TOOLS_EDGE
