from __future__ import annotations

from agent.agentstate.model import AgentState
from agent.graph_constants import EXECUTE_TOOLS_EDGE, SYNTHESIZE_EDGE


def validator(state: AgentState) -> str:
    # Potentially implement this as a fallback i.e a repair node which essentially calls the planner with more context
    #    if plan is None:
    #        return "repair"
    last_iteration = state.iteration_trace[-1]
    if state.goal_reached or last_iteration.plan.final_answer:
        return SYNTHESIZE_EDGE
    # also route to solve if planner produced no steps (if you track that)
    return EXECUTE_TOOLS_EDGE
