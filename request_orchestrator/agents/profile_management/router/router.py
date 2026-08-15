from __future__ import annotations

from langgraph.graph import END

from request_orchestrator.constants import EVALUATE_EDGE, EXECUTE_TOOLS_EDGE, PLAN_EDGE
from request_orchestrator.models.agent_state import AgentState
from request_orchestrator.models.plan_step_ids import format_plan_step_id
from request_orchestrator.shared.planner_state import current_plan_results


def router(state: AgentState) -> str:
    planner_state = state.node_states.planner
    evaluator_state = state.node_states.evaluator
    if planner_state.plan is None:
        return PLAN_EDGE

    plan = planner_state.plan

    if planner_state.plan_count >= state.max_turns:
        return END

    if plan is None or len(plan.steps) == 0:
        return END

    current_results = current_plan_results(
        planner_state,
        state.result.tool_results,
        agent_name=state.agent_profile.name,
    )
    plan_number = planner_state.plan_count
    has_pending_steps = any(
        format_plan_step_id(plan_number, step.id) not in current_results
        for step in plan.steps
    )
    if has_pending_steps:
        return EXECUTE_TOOLS_EDGE

    # if there were tool results and a replan was needed by the planner retrn to planning
    if current_results and planner_state.needs_replan:
        return PLAN_EDGE

    if current_results:
        return EVALUATE_EDGE

    return PLAN_EDGE
