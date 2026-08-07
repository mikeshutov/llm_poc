from __future__ import annotations

from langgraph.graph import END

from request_orchestrator.constants import EXECUTE_TOOLS_EDGE, PLAN_EDGE
from request_orchestrator.models.agent_state import AgentState


def router(state: AgentState) -> str:
    if not state.iteration_trace:
        return PLAN_EDGE

    last_iteration = state.iteration_trace[-1]
    plan = last_iteration.plan

    if len(state.iteration_trace) >= state.max_turns:
        return END

    if plan is None or plan.final_answer or len(plan.steps) == 0:
        return END

    has_pending_steps = any(step.id not in last_iteration.results for step in plan.steps)
    if has_pending_steps:
        return EXECUTE_TOOLS_EDGE

    return PLAN_EDGE
