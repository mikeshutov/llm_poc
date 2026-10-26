from __future__ import annotations

from time import perf_counter
from uuid import UUID

from langsmith import traceable

from common.data import repair_common_json_issues, strip_code_fences
from common.logging import compact_trace_inputs, compact_trace_outputs, record_node_log
from llm.conversation_model_config import EVALUATOR_STAGE, SHARED_MODEL_SCOPE
from llm.usage import record_llm_call
from request_orchestrator.models.agent_prompt import EvidenceStep
from request_orchestrator.constants import NodeName
from request_orchestrator.models.agent_state import AgentState
from request_orchestrator.models.evaluator_event_payload import EvaluatorEventPayload
from request_orchestrator.models.evaluation_result import (
    EVALUATION_STATUS_PARTIAL_BUT_SUFFICIENT,
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
from request_orchestrator.shared.evaluator.prompts.evaluator_prompt import build_evaluator_prompt
from request_orchestrator.shared.evaluator.completion_gate import try_completion_evaluation
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


def _apply_evaluation(
    state: AgentState,
    evaluation: EvaluationResult,
    evidence_steps: list[EvidenceStep],
) -> list[UUID]:
    all_evidence_ids = [
        str(evidence.id)
        for step in evidence_steps
        for evidence in step.evidence
    ]
    deduped_relevant_evidence = _dedupe_evidence_ids(all_evidence_ids)
    state.result = state.result.copy(relevant_evidence_ids=deduped_relevant_evidence)
    state.node_states.evaluator.evaluation_status = evaluation.status
    state.node_states.evaluator.missing_information = evaluation.missing_information
    if evaluation.status in TERMINAL_EVALUATION_STATUSES:
        state.result = state.result.copy(
            result_status=(
                ResultStatus.SUCCESS
                if evaluation.status in {EVALUATION_STATUS_SATISFIED, EVALUATION_STATUS_PARTIAL_BUT_SUFFICIENT}
                else ResultStatus.FAILED
            )
        )
    return deduped_relevant_evidence


@traceable(name="Evaluator Node", process_inputs=compact_trace_inputs, process_outputs=compact_trace_outputs)
def run_evaluator(state: AgentState) -> AgentState:
    execution_context = state.execution_context
    tool_results = state.gather_tool_results()
    evidence_bundle = build_evidence_bundle_from_tool_results(tool_results)
    evidence_steps = build_evidence_steps_from_tool_results(
        tool_results,
        evidence_bundle.evidence_views_by_tool_call_id,
    )
    try:
        # NLI is only a fast-path completion check. A negative/uncertain result
        # must fall through so the LLM can make the authoritative status decision.
        completion_satisfied = try_completion_evaluation(state, evidence_steps)
    except Exception as exc:
        record_node_log(
            roundtrip_id=execution_context.roundtrip_id,
            node_name=NodeName.EVALUATOR,
            stage="completion_gate",
            event_type="completion_evaluation",
            level="warning",
            message="Completion evaluator unavailable; falling back to the LLM evaluator",
            payload={"error": str(exc)},
        )
        completion_satisfied = False
    if completion_satisfied:
        all_evidence_ids = [
            str(evidence.id)
            for step in evidence_steps
            for evidence in step.evidence
        ]
        deduped_relevant_evidence = _dedupe_evidence_ids(all_evidence_ids)
        state.result = state.result.copy(
            relevant_evidence_ids=deduped_relevant_evidence,
            result_status=ResultStatus.SUCCESS,
        )
        state.node_states.evaluator.evaluation_status = EVALUATION_STATUS_SATISFIED
        state.node_states.evaluator.missing_information = []
        record_node_log(
            conversation_id=execution_context.conversation_id,
            roundtrip_id=execution_context.roundtrip_id,
            event_type=EVALUATOR_KIND,
            source=state.agent_profile.name,
            agent_name=state.agent_profile.name,
            node_name=NodeName.EVALUATOR,
            agent_version=state.agent_profile.version,
            payload=EvaluatorEventPayload.from_evaluation(
                agent_name=state.agent_profile.name,
                kind=EVALUATOR_KIND,
                status=EVALUATION_STATUS_SATISFIED,
                relevant_evidence=[str(evidence_id) for evidence_id in deduped_relevant_evidence],
                missing_information=[],
            ).model_dump(),
        )
        return state

    prompt = build_evaluator_prompt(state=state, evidence=evidence_steps)
    prompt_messages = prompt.build_messages()
    prompt_input_object = {"messages": prompt_messages}
    llm = build_llm_for_stage(
        execution_context=execution_context,
        llm=state.llm,
        agent=SHARED_MODEL_SCOPE,
        stage=EVALUATOR_STAGE,
        agent_profile=state.agent_profile,
    )
    started_at = perf_counter()
    response = llm.invoke(prompt_messages)
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
        record_node_log(
            conversation_id=execution_context.conversation_id,
            roundtrip_id=execution_context.roundtrip_id,
            event_type=EVALUATOR_KIND,
            source=state.agent_profile.name,
            agent_name=state.agent_profile.name,
            node_name=NodeName.EVALUATOR,
            agent_version=state.agent_profile.version,
            payload=EvaluatorEventPayload.from_parse_error(
                agent_name=state.agent_profile.name,
                kind=EVALUATOR_KIND,
                status=EVALUATION_STATUS_TERMINAL,
                parse_error=str(exc),
            ).model_dump(),
        )
        return state

    deduped_relevant_evidence = _apply_evaluation(state, evaluation, evidence_steps)
    record_node_log(
        conversation_id=execution_context.conversation_id,
        roundtrip_id=execution_context.roundtrip_id,
        event_type=EVALUATOR_KIND,
        source=state.agent_profile.name,
        agent_name=state.agent_profile.name,
        node_name=NodeName.EVALUATOR,
        agent_version=state.agent_profile.version,
        payload=EvaluatorEventPayload.from_evaluation(
            agent_name=state.agent_profile.name,
            kind=EVALUATOR_KIND,
            status=evaluation.status,
            relevant_evidence=[str(evidence_id) for evidence_id in deduped_relevant_evidence],
            missing_information=evaluation.missing_information,
        ).model_dump(),
    )

    return state
