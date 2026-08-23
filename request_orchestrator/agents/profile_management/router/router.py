from __future__ import annotations

from langgraph.graph import END

from request_orchestrator.constants import EVALUATE_EDGE, EXECUTE_TOOLS_EDGE, PLAN_EDGE
from request_orchestrator.models.agent_state import AgentState
from request_orchestrator.models.plan_step_ids import format_plan_step_id, namespace_step_id


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

    plan_number = planner_state.plan_count
    current_result_step_ids = {
        namespace_step_id(
            state.agent_profile.name,
            format_plan_step_id(plan_number, step.id),
        )
        for step in plan.steps
    }
    current_results = {
        tool_result.step_id: tool_result
        for tool_result in state.gather_tool_results()
        if tool_result.step_id in current_result_step_ids
    }
    has_pending_steps = any(
        namespace_step_id(
            state.agent_profile.name,
            format_plan_step_id(plan_number, step.id),
        ) not in current_results
        for step in plan.steps
    )
    if has_pending_steps:
        return EXECUTE_TOOLS_EDGE

    if current_results:
        # The evaluator owns replanning for this strategy. Do not let a planner
        # flag bypass it, or repeated tool calls can starve evaluation.
        return EVALUATE_EDGE

    return PLAN_EDGE
