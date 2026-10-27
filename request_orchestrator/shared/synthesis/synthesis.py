from __future__ import annotations
from time import perf_counter
from uuid import UUID

from langsmith import traceable

from common.data import repair_common_json_issues, strip_code_fences
from common.logging import compact_trace_inputs, compact_trace_outputs, record_node_log
from llm.conversation_model_config import MAIN_AGENT_MODEL_SCOPE, SYNTHESIS_STAGE
from llm.usage import record_llm_call
from request_orchestrator.models.main_state import MainState
from request_orchestrator.constants import NodeName
from request_orchestrator.models.orchestrator_result import OrchestratorResult
from request_orchestrator.models.synthesized_result import SynthesisResult, SynthesisResultBlock
from request_orchestrator.shared.evidence import (
    build_evidence_bundle_from_tool_results,
    build_evidence_steps_from_tool_results,
    filter_evidence_steps,
)
from llm.chat_models import build_llm_for_stage, resolve_stage_model_name, resolve_stage_provider_name
from request_orchestrator.shared.synthesis.prompts.synthesis_prompt import build_synthesis_prompt
from rendering.debug import SYNTHESIS_KIND
def _resolve_relevant_evidence_ids(state: MainState) -> set[str]:
    return {
        str(evidence_id)
        for evidence_id in [*state.relevant_evidence_ids, *state.gather_relevant_evidence_ids()]
    }


def _resolve_agent_name(state: MainState) -> str:
    return "request_orchestrator"


def _resolve_synthesis_model_name(state: MainState) -> str:
    return resolve_stage_model_name(
        execution_context=state.execution_context,
        agent=MAIN_AGENT_MODEL_SCOPE,
        stage=SYNTHESIS_STAGE,
    )


def _build_result_blocks(synthesis_result: SynthesisResult, evidence_steps) -> tuple[list, list[str]]:
    result_blocks = [block.model_copy(deep=True) for block in synthesis_result.result]
    used_evidence_ids = [
        evidence_id
        for block in result_blocks
        for evidence_id in block.evidence_ids
        if evidence_id
    ]
    if used_evidence_ids:
        return result_blocks, used_evidence_ids

    evidence_ids: list[str] = []
    seen: set[str] = set()
    for step in evidence_steps:
        for evidence in step.evidence:
            evidence_id = str(evidence.id)
            if evidence_id in seen:
                continue
            seen.add(evidence_id)
            evidence_ids.append(evidence_id)

    return (
        [
            block.model_copy(update={"evidence_ids": list(evidence_ids)})
            for block in result_blocks
        ],
        evidence_ids,
    )


@traceable(name="Synthesis Node", process_inputs=compact_trace_inputs, process_outputs=compact_trace_outputs)
def run_synthesis(state: MainState) -> MainState:
    execution_context = state.execution_context
    tool_results = state.gather_tool_results()
    relevant_evidence_ids = _resolve_relevant_evidence_ids(state)
    evidence_bundle = build_evidence_bundle_from_tool_results(tool_results)
    all_evidence_steps = build_evidence_steps_from_tool_results(
        tool_results,
        evidence_bundle.evidence_views_by_tool_call_id,
    )
    evidence_steps = filter_evidence_steps(all_evidence_steps, relevant_evidence_ids)
    if not evidence_steps:
        evidence_steps = all_evidence_steps

    prompt = build_synthesis_prompt(evidence=evidence_steps, state=state)
    prompt_messages = prompt.build_messages()
    llm = build_llm_for_stage(
        execution_context=execution_context,
        llm=state.llm,
        agent=MAIN_AGENT_MODEL_SCOPE,
        stage=SYNTHESIS_STAGE,
    )
    started_at = perf_counter()
    response = llm.invoke(prompt_messages)
    latency_ms = int((perf_counter() - started_at) * 1000)
    llm_call = record_llm_call(
        raw_response=response,
        model_name=_resolve_synthesis_model_name(state),
        provider=resolve_stage_provider_name(
            execution_context=execution_context,
            agent=MAIN_AGENT_MODEL_SCOPE,
            stage=SYNTHESIS_STAGE,
        ),
        conversation_id=execution_context.conversation_id,
        roundtrip_id=execution_context.roundtrip_id,
        user_id=execution_context.user_profile.user_id,
        agent=MAIN_AGENT_MODEL_SCOPE,
        stage=SYNTHESIS_STAGE,
        node_name=NodeName.SYNTHESIS,
        callsite="shared_synthesis.run_synthesis",
        latency_ms=latency_ms,
        owner_agent_name=_resolve_agent_name(state),
        input_object={"messages": prompt_messages},
        output_object={
            "raw_content": response.content,
        },
    )
    raw = repair_common_json_issues(strip_code_fences(response.content))

    try:
        synthesis_result = SynthesisResult.model_validate_json(raw)
    except Exception as e:
        record_node_log(
            conversation_id=execution_context.conversation_id,
            roundtrip_id=execution_context.roundtrip_id,
            node_name=NodeName.SYNTHESIS,
            stage="parse",
            event_type="synthesis_parse_failure",
            level="error",
            message="Synthesis returned non-JSON output; using bounded text fallback",
            payload={
                "error": str(e),
                "raw_preview": raw[:2_000],
                "raw_length": len(raw),
            },
            duration_ms=latency_ms,
        )
        fallback_text = raw.strip() or "I was unable to produce a response."
        synthesis_result = SynthesisResult(
            result=[SynthesisResultBlock(content=fallback_text)],
            roundtrip_summary=fallback_text[:1_000],
        )

    result_blocks, used_evidence_ids = _build_result_blocks(synthesis_result, evidence_steps)

    log_data = {
        "answer_preview": [str(block.content)[:1_000] for block in result_blocks[:3]],
        "next_question_preview": synthesis_result.next_question[:1_000],
        "summary_preview": synthesis_result.roundtrip_summary[:1_000],
        "relevant_evidence_ids": used_evidence_ids,
    }
    record_node_log(
        conversation_id=execution_context.conversation_id,
        roundtrip_id=execution_context.roundtrip_id,
        event_type=SYNTHESIS_KIND,
        source=_resolve_agent_name(state),
        agent_name=_resolve_agent_name(state),
        node_name=NodeName.SYNTHESIS,
        payload={
            "agent_name": _resolve_agent_name(state),
            "kind": SYNTHESIS_KIND,
            "data": log_data,
        },
    )

    state.result = OrchestratorResult(
        agent_result=state.result.agent_result.copy(
            tool_call_ids=state.gather_tool_call_ids(),
            relevant_evidence_ids=[UUID(evidence_id) for evidence_id in relevant_evidence_ids],
        ),
        result_blocks=result_blocks,
        answer=[
            block.content.strip()
            for block in result_blocks
            if isinstance(block.content, str) and block.content.strip()
        ],
        next_question=synthesis_result.next_question.strip(),
        roundtrip_summary=synthesis_result.roundtrip_summary.strip(),
        roundtrip_latency_ms=state.result.roundtrip_latency_ms,
    )

    return state

