from __future__ import annotations

from langgraph.graph import END

from common.logging import record_node_log
from rendering.debug import EXECUTION_RESULT_KIND
from agent_runtime.constants import EVALUATE_EDGE, PLAN_EDGE, SYNTHESIZE_EDGE, NodeName
from agent_runtime.models.agent_result import AnswerStatus, ExecutionStatus
from agent_runtime.models.agent_state import AgentState


def run_execution_result_validator(state: AgentState) -> AgentState:
    plan = state.node_states.planner.plan
    if plan is None:
        state.result = state.result.copy(
            answer_status=AnswerStatus.FAILED,
            execution_status=ExecutionStatus.FAILED,
        )
        return state
    step_ids = {step.db_id for step in plan.steps}
    calls = [call for call in state.gather_tool_calls() if call.plan_step_id in step_ids]
    tool_results = [
        tool_result
        for tool_result in state.gather_tool_results()
        if tool_result.plan_step_id in step_ids
    ]
    statuses = [call.status for call in calls]
    if statuses and all(status == "rejected" for status in statuses):
        answer_status = AnswerStatus.FAILED
        execution_status = ExecutionStatus.NO_NEW_WORK
    else:
        has_evidence = any(tool_result.evidence for tool_result in tool_results)
        answer_status = AnswerStatus.PARTIAL_SUCCESS if has_evidence else AnswerStatus.FAILED
        completed_count = sum(call_status == "completed" for call_status in statuses)
        if completed_count == len(statuses) and statuses:
            execution_status = ExecutionStatus.SUCCESS
        elif completed_count:
            execution_status = ExecutionStatus.PARTIAL
        else:
            execution_status = ExecutionStatus.FAILED
    state.result = state.result.copy(
        answer_status=answer_status,
        execution_status=execution_status,
    )
    if answer_status is AnswerStatus.FAILED:
        state.node_states.planner.no_result_attempts += 1
    else:
        state.node_states.planner.no_result_attempts = 0
    return state


def execution_result_router(state: AgentState) -> str:
    if state.result.execution_status is ExecutionStatus.NO_NEW_WORK:
        plan = state.node_states.planner.plan
        rejected_call_count = len(plan.steps) if plan else 0
        record_node_log(
            conversation_id=state.execution_context.conversation_id,
            roundtrip_id=state.execution_context.roundtrip_id,
            event_type=EXECUTION_RESULT_KIND,
            source=state.agent_profile.name,
            agent_name=state.agent_profile.name,
            agent_version=state.agent_profile.version,
            node_name=NodeName.VALIDATE_EXECUTION_RESULT,
            payload={"agent_name": state.agent_profile.name, "kind": EXECUTION_RESULT_KIND, "data": {
                "title": "Execution Stopped Early",
                "status": ExecutionStatus.NO_NEW_WORK,
                "reason": "All planned tool requests were already completed in this roundtrip.",
                "rejected_call_count": rejected_call_count,
            }},
        )
        return SYNTHESIZE_EDGE
    if state.result.answer_status is AnswerStatus.PARTIAL_SUCCESS:
        return EVALUATE_EDGE
    if state.node_states.planner.no_result_attempts == 1:
        return PLAN_EDGE
    return END
