from __future__ import annotations

from langsmith import traceable

from common.parsing import strip_code_fences
from conversation.models.conversation_model_config import EVALUATOR_STAGE, SHARED_MODEL_SCOPE
from conversation.repository.repo_factory import get_conversation_repo
from llm.usage import record_llm_call, serialize_llm_call_record
from request_orchestrator.constants import EVALUATOR_PROMPT_STEP
from request_orchestrator.models.agent_prompt import PlanEvidenceStep
from request_orchestrator.models.evaluation_result import EvaluationResult
from request_orchestrator.models.agent_state import AgentState
from request_orchestrator.shared.evaluator.prompts import build_evaluator_prompt
from request_orchestrator.shared.prompts.render_agent_prompt import render_agent_prompt

EVALUATOR_KIND = "evaluator"


def _build_plan_with_evidence(state: AgentState) -> list[PlanEvidenceStep]:
    plan_with_evidence: list[PlanEvidenceStep] = []
    for iteration in state.iteration_trace:
        if iteration.plan is None:
            continue
        for step in iteration.plan.steps:
            plan_with_evidence.append(
                PlanEvidenceStep(
                    step_id=step.id,
                    plan=step.plan,
                    tool=step.tool,
                    args=step.args,
                    evidence=iteration.results.get(step.id, ""),
                )
            )
    return plan_with_evidence


@traceable(name="Evaluator Node")
def run_evaluator(state: AgentState) -> AgentState:
    plan_with_evidence = _build_plan_with_evidence(state)
    prompt = build_evaluator_prompt(state=state, plan_with_evidence=plan_with_evidence)
    prompt_text = render_agent_prompt(prompt)
    llm = state.build_llm_for_stage(agent=SHARED_MODEL_SCOPE, stage=EVALUATOR_STAGE)
    response = llm.invoke(prompt_text)
    llm_call = record_llm_call(
        raw_response=response,
        model_name=state.resolve_model_for_stage(agent=SHARED_MODEL_SCOPE, stage=EVALUATOR_STAGE),
        conversation_id=state.conversation_id,
        roundtrip_id=state.roundtrip_id,
        agent=SHARED_MODEL_SCOPE,
        stage=EVALUATOR_STAGE,
        callsite="shared_evaluator.run_evaluator",
        metadata={"evidence_count": len(plan_with_evidence)},
        input_object={
            "prompt": prompt_text,
        },
        output_object={
            "raw_content": response.content,
        },
    )
    raw = strip_code_fences(response.content)

    try:
        evaluation = EvaluationResult.model_validate_json(raw)
    except Exception as exc:
        state.goal_reached = True
        state.relevant_evidence_ids = []
        state.log_status(
            agent_name=state.agent_profile.name,
            kind=EVALUATOR_KIND,
            data={
                "satisfied": True,
                "relevant_evidence": [],
                "missing_information": [],
                "refined_goal": "",
                "parse_error": str(exc),
                "llm_usage": None if llm_call is None else serialize_llm_call_record(llm_call),
            },
        )
        return state

    state.relevant_evidence_ids = list(evaluation.relevant_evidence)

    if evaluation.satisfied:
        state.goal_reached = True
    else:
        refined_goal = evaluation.refined_goal.strip()
        if refined_goal:
            state.request_analysis.goal = refined_goal
        state.goal_reached = False

    state.log_status(
        agent_name=state.agent_profile.name,
        kind=EVALUATOR_KIND,
        data={
            "satisfied": evaluation.satisfied,
            "relevant_evidence": evaluation.relevant_evidence,
            "missing_information": evaluation.missing_information,
            "refined_goal": evaluation.refined_goal,
            "llm_usage": None if llm_call is None else serialize_llm_call_record(llm_call),
        },
    )

    if state.roundtrip_id:
        get_conversation_repo().create_roundtrip_prompt(
            state.roundtrip_id,
            agent=state.agent_profile.name,
            prompt_step=EVALUATOR_PROMPT_STEP,
            prompt=prompt_text,
        )

    return state
