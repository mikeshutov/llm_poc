from __future__ import annotations

from time import perf_counter

from langsmith import traceable

from common.data import repair_common_json_issues, strip_code_fences
from common.logging import log_roundtrip_prompt
from conversation.models.conversation_model_config import EVALUATOR_STAGE, SHARED_MODEL_SCOPE
from llm.usage import record_llm_call, serialize_llm_call_record
from request_orchestrator.constants import EVALUATOR_PROMPT_STEP
from request_orchestrator.models.agent_state import AgentState
from request_orchestrator.models.evaluation_result import (
    EVALUATION_STATUS_TERMINAL,
    EvaluationResult,
    TERMINAL_EVALUATION_STATUSES,
)
from request_orchestrator.shared.evidence import build_evidence_bundle, build_evidence_steps
from request_orchestrator.shared.evaluator.prompts import build_evaluator_prompt

EVALUATOR_KIND = "evaluator"


def _dedupe_string_list(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


@traceable(name="Evaluator Node")
def run_evaluator(state: AgentState) -> AgentState:
    evidence_bundle = build_evidence_bundle(state.iteration_trace)
    evidence_steps = build_evidence_steps(
        state.iteration_trace,
        evidence_bundle.evidence_views_by_step_id,
    )
    prompt = build_evaluator_prompt(state=state, evidence=evidence_steps)
    prompt_text = prompt.prompt_text()
    prompt_input_object = prompt.to_log_input_object()
    llm = state.build_llm_for_stage(agent=SHARED_MODEL_SCOPE, stage=EVALUATOR_STAGE)
    started_at = perf_counter()
    response = llm.invoke(prompt_text)
    latency_ms = int((perf_counter() - started_at) * 1000)
    llm_call = record_llm_call(
        raw_response=response,
        model_name=state.resolve_model_for_stage(agent=SHARED_MODEL_SCOPE, stage=EVALUATOR_STAGE),
        conversation_id=state.conversation_id,
        roundtrip_id=state.roundtrip_id,
        user_id=state.user_profile.user_id,
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
        state.evaluation_status = EVALUATION_STATUS_TERMINAL
        state.goal_reached = True
        state.relevant_evidence_ids = []
        state.log_status(
            agent_name=state.agent_profile.name,
            kind=EVALUATOR_KIND,
            status=EVALUATION_STATUS_TERMINAL,
            data={
                "status": EVALUATION_STATUS_TERMINAL,
                "relevant_evidence": [],
                "missing_information": [],
                "refined_goal": "",
                "parse_error": str(exc),
                "llm_usage": None if llm_call is None else serialize_llm_call_record(llm_call),
            },
        )
        return state

    deduped_relevant_evidence = _dedupe_string_list(evaluation.relevant_evidence)
    state.relevant_evidence_ids = deduped_relevant_evidence
    state.evaluation_status = evaluation.status

    if evaluation.status in TERMINAL_EVALUATION_STATUSES:
        state.goal_reached = True
    else:
        refined_goal = evaluation.refined_goal.strip()
        if refined_goal:
            state.request_analysis.set_goal_for_agent(state.agent_profile.name, refined_goal)
        state.goal_reached = False

    state.log_status(
        agent_name=state.agent_profile.name,
        kind=EVALUATOR_KIND,
        status=evaluation.status,
        data={
            "status": evaluation.status,
            "relevant_evidence": deduped_relevant_evidence,
            "missing_information": evaluation.missing_information,
            "refined_goal": evaluation.refined_goal,
            "llm_usage": None if llm_call is None else serialize_llm_call_record(llm_call),
        },
    )

    if state.roundtrip_id:
        log_roundtrip_prompt(
            roundtrip_id=state.roundtrip_id,
            agent=state.agent_profile.name,
            prompt_step=EVALUATOR_PROMPT_STEP,
            prompt=prompt_text,
        )

    return state
