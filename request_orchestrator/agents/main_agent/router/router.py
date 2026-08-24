from __future__ import annotations

from request_orchestrator.models.agent_state import AgentState
from request_orchestrator.constants import EVALUATE_EDGE, PLAN_EDGE, SYNTHESIZE_EDGE


def router(state: AgentState) -> str:
    planner_state = state.node_states.planner
    evaluator_state = state.node_states.evaluator
    if evaluator_state.goal_reached:
        return SYNTHESIZE_EDGE

    if planner_state.plan_count >= state.max_turns:
        return SYNTHESIZE_EDGE

    current_result_step_ids = {
        step.db_id for step in planner_state.plan.steps
    } if planner_state.plan is not None else set()
    current_results = [
        tool_result
        for tool_result in state.gather_tool_results()
        if tool_result.plan_step_id in current_result_step_ids
    ]
    if current_results:
        # The evaluator owns replanning for this strategy. Do not let a planner
        # flag bypass it, or repeated tool calls can starve evaluation.
        return EVALUATE_EDGE

    return PLAN_EDGE
