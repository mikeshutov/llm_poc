from __future__ import annotations

from request_orchestrator.constants import EVALUATE_EDGE, PLAN_EDGE, SYNTHESIZE_EDGE
from request_orchestrator.models.agent_state import AgentState
from request_orchestrator.models.evaluation_result import TERMINAL_EVALUATION_STATUSES


def router(state: AgentState) -> str:
    planner_state = state.node_states.planner
    evaluator_state = state.node_states.evaluator
    if evaluator_state.evaluation_status in TERMINAL_EVALUATION_STATUSES:
        return SYNTHESIZE_EDGE

    if planner_state.plan_count >= state.max_turns:
        return SYNTHESIZE_EDGE

    if planner_state.tool_calls:
        current_plan_id = planner_state.plan.db_id if planner_state.plan else None
        current_results = [
            tool_result
            for tool_result in state.gather_tool_results()
            if tool_result.plan_id == current_plan_id
        ]
    else:
        current_results = []
    if current_results:
        # The evaluator owns replanning for this strategy. Do not let a planner
        # flag bypass it, or repeated tool calls can starve evaluation.
        return EVALUATE_EDGE

    return PLAN_EDGE
