from __future__ import annotations

from request_orchestrator.models.agent_state import AgentState
from request_orchestrator.constants import EVALUATE_EDGE, PLAN_EDGE, SYNTHESIZE_EDGE
from request_orchestrator.shared.planner_state import current_plan_results


def router(state: AgentState) -> str:
    planner_state = state.node_states.planner
    evaluator_state = state.node_states.evaluator
    if evaluator_state.goal_reached:
        return SYNTHESIZE_EDGE

    if planner_state.plan_count >= state.max_turns:
        return SYNTHESIZE_EDGE

    current_results = current_plan_results(
        planner_state,
        state.result.tool_results,
        agent_name=state.agent_profile.name,
    )
    if current_results and planner_state.needs_replan:
        return PLAN_EDGE
    if current_results:
        return EVALUATE_EDGE

    return PLAN_EDGE
