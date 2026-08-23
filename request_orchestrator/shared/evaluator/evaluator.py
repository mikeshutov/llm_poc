from __future__ import annotations

from time import perf_counter
from uuid import UUID

from langsmith import traceable

from common.data import repair_common_json_issues, strip_code_fences
from common.logging import create_conversation_event, log_roundtrip_prompt
from llm.conversation_model_config import EVALUATOR_STAGE, SHARED_MODEL_SCOPE
from llm.usage import record_llm_call
from request_orchestrator.constants import EVALUATOR_PROMPT_KIND
from request_orchestrator.models.agent_state import AgentState
from request_orchestrator.models.evaluator_event_payload import EvaluatorEventPayload
from request_orchestrator.models.evaluation_result import (
    EVALUATION_STATUS_TERMINAL,
    EVALUATION_STATUS_SATISFIED,
    EvaluationResult,
    TERMINAL_EVALUATION_STATUSES,
)
from request_orchestrator.models.agent_result import ResultStatus
from request_orchestrator.shared.evidence import (
    build_evidence_bundle_from_tool_results,
    build_evidence_steps_from_tool_results,
)
from request_orchestrator.shared.evaluator.prompts import build_evaluator_prompt
from llm.chat_models import build_llm_for_stage, resolve_stage_model_name, resolve_stage_provider_name

EVALUATOR_KIND = "evaluator"


def _dedupe_evidence_ids(values: list[str]) -> list[UUID]:
    seen: set[UUID] = set()
    deduped: list[UUID] = []
    for value in values:
        try:
            evidence_id = UUID(value)
        except (TypeError, ValueError):
            continue
        if evidence_id in seen:
            continue
        seen.add(evidence_id)
        deduped.append(evidence_id)
    return deduped


@traceable(name="Evaluator Node")
def run_evaluator(state: AgentState) -> AgentState:
    execution_context = state.execution_context
    tool_results = state.gather_tool_results()
    evidence_bundle = build_evidence_bundle_from_tool_results(tool_results)
    evidence_steps = build_evidence_steps_from_tool_results(
        tool_results,
        evidence_bundle.evidence_views_by_tool_call_id,
    )
    prompt = build_evaluator_prompt(state=state, evidence=evidence_steps)
    prompt_text = prompt.build()
    prompt_input_object = prompt.to_log_input_object()
    llm = build_llm_for_stage(
        execution_context=execution_context,
        llm=state.llm,
        agent=SHARED_MODEL_SCOPE,
        stage=EVALUATOR_STAGE,
        agent_profile=state.agent_profile,
    )
    started_at = perf_counter()
    response = llm.invoke(prompt_text)
    latency_ms = int((perf_counter() - started_at) * 1000)
    llm_call = record_llm_call(
        raw_response=response,
        model_name=resolve_stage_model_name(
            execution_context=execution_context,
            agent=SHARED_MODEL_SCOPE,
            stage=EVALUATOR_STAGE,
            agent_profile=state.agent_profile,
        ),
        provider=resolve_stage_provider_name(
            execution_context=execution_context,
            agent=SHARED_MODEL_SCOPE,
            stage=EVALUATOR_STAGE,
            agent_profile=state.agent_profile,
        ),
        conversation_id=execution_context.conversation_id,
        roundtrip_id=execution_context.roundtrip_id,
        user_id=execution_context.user_profile.user_id,
        agent=SHARED_MODEL_SCOPE,
        stage=EVALUATOR_STAGE,
        callsite="shared_evaluator.run_evaluator",
        metadata={"evidence_count": len(evidence_steps)},
        latency_ms=latency_ms,
        owner_agent_name=state.agent_profile.name,
        input_object=prompt_input_object,
        output_object={
            "raw_content": response.content,
        },
    )
    raw = repair_common_json_issues(strip_code_fences(response.content))

    try:
        evaluation = EvaluationResult.model_validate_json(raw)
    except Exception as exc:
        state.result = state.result.copy(relevant_evidence_ids=[], result_status=ResultStatus.FAILED)
        state.node_states.evaluator.evaluation_status = EVALUATION_STATUS_TERMINAL
        state.node_states.evaluator.goal_reached = True
        create_conversation_event(
            conversation_id=execution_context.conversation_id,
            roundtrip_id=execution_context.roundtrip_id,
            event_type=EVALUATOR_KIND,
            source=state.agent_profile.name,
            agent_name=state.agent_profile.name,
            payload=EvaluatorEventPayload.from_parse_error(
                agent_name=state.agent_profile.name,
                kind=EVALUATOR_KIND,
                status=EVALUATION_STATUS_TERMINAL,
                parse_error=str(exc),
                llm_call=llm_call,
            ).model_dump(),
        )
        return state

    deduped_relevant_evidence = _dedupe_evidence_ids(evaluation.relevant_evidence)
    state.result = state.result.copy(relevant_evidence_ids=deduped_relevant_evidence)
    state.node_states.evaluator.evaluation_status = evaluation.status

    if evaluation.status in TERMINAL_EVALUATION_STATUSES:
        state.node_states.evaluator.goal_reached = True
        state.result = state.result.copy(
            result_status=(
                ResultStatus.SUCCESS
                if evaluation.status == EVALUATION_STATUS_SATISFIED
                else ResultStatus.FAILED
            )
        )
    else:
        refined_goal = evaluation.refined_goal.strip()
        if refined_goal:
            state.inputs.task = refined_goal
        state.node_states.evaluator.goal_reached = False

    create_conversation_event(
        conversation_id=execution_context.conversation_id,
        roundtrip_id=execution_context.roundtrip_id,
        event_type=EVALUATOR_KIND,
        source=state.agent_profile.name,
        agent_name=state.agent_profile.name,
        payload=EvaluatorEventPayload.from_evaluation(
            agent_name=state.agent_profile.name,
            kind=EVALUATOR_KIND,
            status=evaluation.status,
            relevant_evidence=[str(evidence_id) for evidence_id in deduped_relevant_evidence],
            missing_information=evaluation.missing_information,
            refined_goal=evaluation.refined_goal,
            llm_call=llm_call,
        ).model_dump(),
    )

    if execution_context.roundtrip_id:
        log_roundtrip_prompt(
            roundtrip_id=execution_context.roundtrip_id,
            agent=state.agent_profile.name,
            prompt_step=EVALUATOR_PROMPT_KIND,
            prompt=prompt_text,
        )

    return state
