from __future__ import annotations

from agent_runtime.constants import PLAN_EDGE, SYNTHESIZE_EDGE
from agent_runtime.models.agent_state import AgentState
from agent_runtime.models.evaluation_result import TERMINAL_EVALUATION_STATUSES


def _missing_completion_tools(state: AgentState) -> set[str]:
    required_tools = state.agent_profile.required_completion_tools
    if not required_tools:
        return set()

    completed_tools: set[str] = set()
    for result in state.gather_tool_results():
        if not result.tool_name or (
            isinstance(result.result, dict) and result.result.get("error")
        ):
            continue
        completed_tools.add(result.tool_name)
    return required_tools - completed_tools


def evaluator_router(state: AgentState) -> str:
    planner_state = state.node_states.planner
    evaluator_state = state.node_states.evaluator
    if planner_state.plan_count >= state.max_turns:
        return SYNTHESIZE_EDGE
    if evaluator_state.evaluation_status in TERMINAL_EVALUATION_STATUSES:
        if _missing_completion_tools(state):
            return PLAN_EDGE
        return SYNTHESIZE_EDGE
    return PLAN_EDGE
