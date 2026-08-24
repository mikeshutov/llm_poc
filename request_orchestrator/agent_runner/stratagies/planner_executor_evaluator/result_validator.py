from __future__ import annotations

from request_orchestrator.constants import EVALUATE_EDGE, PLAN_EDGE, SYNTHESIZE_EDGE
from request_orchestrator.models.agent_result import ResultStatus
from request_orchestrator.models.agent_state import AgentState
from common.logging import create_conversation_event
from rendering.debug import EXECUTION_RESULT_KIND


def run_execution_result_validator(state: AgentState) -> AgentState:
    plan = state.node_states.planner.plan
    if plan is None:
        state.result = state.result.copy(result_status=ResultStatus.FAILED)
        return state

    step_ids = {step.db_id for step in plan.steps}
    calls = [call for call in state.gather_tool_calls() if call.plan_step_id in step_ids]
    statuses = [call.status for call in calls]
    if statuses and all(status == "rejected" for status in statuses):
        status = ResultStatus.NO_NEW_WORK
    elif any(status == "completed" for status in statuses):
        status = ResultStatus.PARTIAL_SUCCESS
    else:
        status = ResultStatus.FAILED
    state.result = state.result.copy(result_status=status)
    return state


def execution_result_router(state: AgentState) -> str:
    if state.result.result_status is ResultStatus.NO_NEW_WORK:
        plan = state.node_states.planner.plan
        rejected_call_count = len(plan.steps) if plan is not None else 0
        create_conversation_event(
            conversation_id=state.execution_context.conversation_id,
            roundtrip_id=state.execution_context.roundtrip_id,
            event_type=EXECUTION_RESULT_KIND,
            source=state.agent_profile.name,
            agent_name=state.agent_profile.name,
            node_name="execution_result_validator",
            payload={
                "agent_name": state.agent_profile.name,
                "kind": EXECUTION_RESULT_KIND,
                "data": {
                    "title": "Execution Stopped Early",
                    "status": ResultStatus.NO_NEW_WORK,
                    "reason": "All planned tool requests were already completed in this roundtrip.",
                    "rejected_call_count": rejected_call_count,
                },
            },
        )
        return SYNTHESIZE_EDGE
    if state.result.result_status is ResultStatus.PARTIAL_SUCCESS:
        return EVALUATE_EDGE
    return PLAN_EDGE
