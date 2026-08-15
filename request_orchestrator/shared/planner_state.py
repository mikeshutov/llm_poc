from __future__ import annotations

from dataclasses import dataclass

from request_orchestrator.models.plan import Plan
from request_orchestrator.models.plan_step_ids import format_plan_step_id, namespace_step_id


@dataclass
class PlannerNodeState:
    node_name: str = "planner"
    plan: Plan | None = None
    needs_replan: bool = False
    plan_count: int = 0

def current_plan_results(
    planner_state: PlannerNodeState,
    tool_results: list["ToolResult"],
    *,
    agent_name: str,
) -> dict[str, "ToolResult"]:
    if planner_state.plan is None:
        return {}
    local_step_ids = {
        format_plan_step_id(planner_state.plan_count, step.id)
        for step in planner_state.plan.steps
    }
    namespaced_step_ids_by_local = {
        step_id: namespace_step_id(agent_name, step_id)
        for step_id in local_step_ids
    }
    return {
        local_step_id: tool_result
        for tool_result in tool_results
        for local_step_id, namespaced_step_id in namespaced_step_ids_by_local.items()
        if tool_result.step_id == namespaced_step_id
    }
