from __future__ import annotations

from time import perf_counter

from langsmith import traceable

from common.data import repair_common_json_issues, strip_code_fences
from conversation.models.conversation_model_config import MAIN_AGENT_MODEL_SCOPE, SYNTHESIS_STAGE
from conversation.repository.repo_factory import get_conversation_repo
from llm.usage import record_llm_call, serialize_llm_call_record
from request_orchestrator.constants import SYNTHESIS_PROMPT_STEP
from request_orchestrator.models.agent_result import AgentResult
from request_orchestrator.models.agent_state import AgentState
from request_orchestrator.models.synthesized_result import SynthesisResult
from request_orchestrator.shared.evidence import (
    build_evidence_bundle,
    build_evidence_steps,
    filter_evidence_steps,
)
from request_orchestrator.shared.synthesis.prompts.solver_prompt import build_solver_prompt
from rendering.debug import SYNTHESIS_KIND


def _resolve_relevant_evidence_ids(state: AgentState) -> set[str]:
    return {
        evidence_id
        for evidence_id in state.relevant_evidence_ids
        if isinstance(evidence_id, str) and evidence_id.strip()
    }


@traceable(name="Synthesis Node")
def run_synthesis(state: AgentState) -> AgentState:
    if not state.iteration_trace and not state.goal_reached:
        state.result = AgentResult(answer=[])
        state.goal_reached = True
        return state

    relevant_evidence_ids = _resolve_relevant_evidence_ids(state)
    evidence_bundle = build_evidence_bundle(state.iteration_trace)
    all_evidence_steps = build_evidence_steps(
        state.iteration_trace,
        evidence_bundle.evidence_views_by_step_id,
    )
    evidence_steps = filter_evidence_steps(all_evidence_steps, relevant_evidence_ids)
    if not evidence_steps:
        evidence_steps = all_evidence_steps

    prompt = build_solver_prompt(evidence=evidence_steps, state=state)
    prompt_text = prompt.prompt_text()
    prompt_input_object = prompt.to_log_input_object()
    llm = state.build_llm_for_stage(
        agent=MAIN_AGENT_MODEL_SCOPE,
        stage=SYNTHESIS_STAGE,
    )
    started_at = perf_counter()
    response = llm.invoke(prompt_text)
    latency_ms = int((perf_counter() - started_at) * 1000)
    llm_call = record_llm_call(
        raw_response=response,
        model_name=state.resolve_model_for_stage(agent=MAIN_AGENT_MODEL_SCOPE, stage=SYNTHESIS_STAGE),
        conversation_id=state.conversation_id,
        roundtrip_id=state.roundtrip_id,
        user_id=state.user_profile.user_id,
        agent=MAIN_AGENT_MODEL_SCOPE,
        stage=SYNTHESIS_STAGE,
        callsite="shared_synthesis.run_synthesis",
        latency_ms=latency_ms,
        input_object=prompt_input_object,
        output_object={
            "raw_content": response.content,
        },
    )
    raw = repair_common_json_issues(strip_code_fences(response.content))

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
    used_evidence_ids = [
        evidence_id
        for block in synthesis_result.result
        for evidence_id in block.evidence_ids
        if evidence_id
    ]
    if not used_evidence_ids:
        used_evidence_ids = [
            evidence.evidence_id
            for step in evidence_steps
            for evidence in step.evidence
        ]

    state.log_status(
        agent_name=state.agent_profile.name,
        kind=SYNTHESIS_KIND,
        data={
            "answer_preview": [block.content for block in synthesis_result.result[:3]],
            "next_question": synthesis_result.next_question,
            "relevant_evidence_ids": used_evidence_ids,
            "llm_usage": None if llm_call is None else serialize_llm_call_record(llm_call),
        },
    )

    state.result = AgentResult.from_state(
        state=state,
        answer_blocks=synthesis_result.result,
        next_question=synthesis_result.next_question,
        roundtrip_summary=synthesis_result.roundtrip_summary,
        tool_summary=tool_summary,
        used_evidence_ids=used_evidence_ids,
        hydrated_evidence_by_id=evidence_bundle.hydrated_evidence_by_id,
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

