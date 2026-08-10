from __future__ import annotations

from langsmith import traceable

from common.parsing import strip_code_fences
from conversation.models.conversation_model_config import MAIN_AGENT_MODEL_SCOPE, SYNTHESIS_STAGE
from conversation.repository.repo_factory import get_conversation_repo
from llm.usage import record_llm_call, serialize_llm_call_record
from request_orchestrator.constants import SYNTHESIS_PROMPT_STEP
from request_orchestrator.models.agent_prompt import PlanEvidenceStep
from request_orchestrator.models.agent_result import AgentResult
from request_orchestrator.models.agent_state import AgentState
from request_orchestrator.models.synthesized_result import SynthesisResult
from request_orchestrator.shared.prompts.render_agent_prompt import render_agent_prompt
from request_orchestrator.shared.synthesis.prompts.solver_prompt import build_solver_prompt
from rendering.debug import SYNTHESIS_KIND


def _resolve_relevant_evidence_ids(state: AgentState) -> set[str]:
    return {step_id for step_id in state.relevant_evidence_ids if isinstance(step_id, str) and step_id.strip()}


@traceable(name="Synthesis Node")
def run_synthesis(state: AgentState) -> AgentState:
    if not state.iteration_trace and not state.goal_reached:
        state.result = AgentResult(answer=[])
        state.goal_reached = True
        return state

    relevant_evidence_ids = _resolve_relevant_evidence_ids(state)
    all_plan_with_evidence: list[PlanEvidenceStep] = []
    for iteration in state.iteration_trace:
        if iteration.plan is None:
            continue

        for step in iteration.plan.steps:
            all_plan_with_evidence.append(
                PlanEvidenceStep(
                    step_id=step.id,
                    plan=step.plan,
                    tool=step.tool,
                    args=step.args,
                    evidence=iteration.results.get(step.id, ""),
                )
            )

    if relevant_evidence_ids:
        plan_with_evidence = [step for step in all_plan_with_evidence if step.step_id in relevant_evidence_ids]
        if not plan_with_evidence:
            plan_with_evidence = all_plan_with_evidence
    else:
        plan_with_evidence = all_plan_with_evidence

    prompt = build_solver_prompt(plan_with_evidence=plan_with_evidence, state=state)
    prompt_text = render_agent_prompt(prompt)
    llm = state.build_llm_for_stage(
        agent=MAIN_AGENT_MODEL_SCOPE,
        stage=SYNTHESIS_STAGE,
    )
    response = llm.invoke(prompt_text)
    llm_call = record_llm_call(
        raw_response=response,
        model_name=state.resolve_model_for_stage(agent=MAIN_AGENT_MODEL_SCOPE, stage=SYNTHESIS_STAGE),
        conversation_id=state.conversation_id,
        roundtrip_id=state.roundtrip_id,
        user_id=state.user_profile.user_id,
        agent=MAIN_AGENT_MODEL_SCOPE,
        stage=SYNTHESIS_STAGE,
        callsite="shared_synthesis.run_synthesis",
        input_object={
            "prompt": prompt_text,
        },
        output_object={
            "raw_content": response.content,
        },
    )
    raw = strip_code_fences(response.content)

    try:
        synthesis_result = SynthesisResult.model_validate_json(raw)
    except Exception as e:
        state.result = AgentResult(
            answer=[f"Synthesis produced invalid JSON result: {e}\nRaw:\n{raw}"]
        )
        state.goal_reached = True
        return state

    had_tool_results = any(bool(iteration.results) for iteration in state.iteration_trace)
    tool_summary = synthesis_result.tool_summary.model_dump() if had_tool_results else {}

    state.log_status(
        agent_name=state.agent_profile.name,
        kind=SYNTHESIS_KIND,
        data={
            "answer_preview": synthesis_result.result[:3],
            "follow_up": synthesis_result.follow_up,
            "clarifying_question": synthesis_result.clarifying_question,
            "relevant_evidence_ids": [step.step_id for step in plan_with_evidence],
            "llm_usage": None if llm_call is None else serialize_llm_call_record(llm_call),
        },
    )

    state.result = AgentResult.from_state(
        state=state,
        answer=synthesis_result.result,
        follow_up=synthesis_result.follow_up,
        clarifying_question=synthesis_result.clarifying_question,
        roundtrip_summary=synthesis_result.roundtrip_summary,
        tool_summary=tool_summary,
    )
    state.goal_reached = True

    if state.roundtrip_id:
        get_conversation_repo().create_roundtrip_prompt(
            state.roundtrip_id,
            agent=state.agent_profile.name,
            prompt_step=SYNTHESIS_PROMPT_STEP,
            prompt=prompt_text,
        )

    return state

