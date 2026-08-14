from __future__ import annotations
from time import perf_counter
from datetime import datetime
from zoneinfo import ZoneInfo

from langsmith import traceable
from langchain_openai import ChatOpenAI

from common.data import repair_common_json_issues, strip_code_fences
from common.logging import create_conversation_event, log_roundtrip_prompt
from conversation.models.conversation_model_config import MAIN_AGENT_MODEL_SCOPE, SYNTHESIS_STAGE
from llm.usage import record_llm_call, serialize_llm_call_record
from request_orchestrator.constants import SYNTHESIS_PROMPT_STEP
from request_orchestrator.models.agent_result import AgentResult
from request_orchestrator.models.agent_state import AgentState
from request_orchestrator.models.main_state import MainState
from request_orchestrator.models.plan_step_ids import format_plan_step_id
from request_orchestrator.models.synthesized_result import SynthesisResult
from request_orchestrator.shared.evidence import (
    build_evidence_bundle,
    build_evidence_steps,
    filter_evidence_steps,
)
from request_orchestrator.shared.synthesis.prompts.synthesis_prompt import build_synthesis_prompt
from rendering.debug import SYNTHESIS_KIND


def _resolve_iteration_trace(state: AgentState | MainState):
    if isinstance(state, MainState):
        return state.gather_iteration_trace()
    return state.iteration_trace


def _resolve_relevant_evidence_ids(state: AgentState | MainState) -> set[str]:
    values = state.gather_relevant_evidence_ids() if isinstance(state, MainState) else state.relevant_evidence_ids
    return {
        evidence_id
        for evidence_id in values
        if isinstance(evidence_id, str) and evidence_id.strip()
    }


def _resolve_agent_name(state: AgentState | MainState) -> str:
    if isinstance(state, MainState):
        return "request_orchestrator"
    return state.agent_profile.name


def _resolve_tool_summary_freshness(state: AgentState | MainState) -> str:
    timezone = "America/Toronto"
    geometadata = getattr(state.user_profile, "geometadata", None)
    if geometadata is not None and isinstance(geometadata.timezone, str) and geometadata.timezone.strip():
        timezone = geometadata.timezone.strip()
    current_date = datetime.now(ZoneInfo(timezone)).date().isoformat()
    return f"current as of {current_date}"


def _resolve_used_tools(iteration_trace) -> list[str]:
    used_tools: list[str] = []
    seen: set[str] = set()
    for iteration_number, iteration in enumerate(iteration_trace, start=1):
        if iteration.plan is None:
            continue
        for step in iteration.plan.steps:
            step_id = format_plan_step_id(iteration_number, step.id)
            if step_id not in iteration.results:
                continue
            tool_name = step.tool.strip()
            if not tool_name or tool_name in seen:
                continue
            seen.add(tool_name)
            used_tools.append(tool_name)
    return used_tools


def _resolve_tool_summary_used_tools(state: AgentState | MainState) -> list[str]:
    if isinstance(state, MainState):
        if state.agent_states:
            return _resolve_used_tools(state.agent_states[-1].iteration_trace)
        return []
    return _resolve_used_tools(state.iteration_trace)


def _resolve_synthesis_model_name(state: AgentState | MainState) -> str:
    if isinstance(state, MainState):
        return state.conversation_model_config.resolve(
            agent=MAIN_AGENT_MODEL_SCOPE,
            stage=SYNTHESIS_STAGE,
        )
    return state.resolve_model_for_stage(agent=MAIN_AGENT_MODEL_SCOPE, stage=SYNTHESIS_STAGE)


def _build_synthesis_llm(state: AgentState | MainState):
    if isinstance(state, MainState):
        model_name = _resolve_synthesis_model_name(state)
        if state.llm is None:
            return ChatOpenAI(model=model_name)
        if not isinstance(state.llm, ChatOpenAI):
            return state.llm
        return ChatOpenAI(model=model_name)
    return state.build_llm_for_stage(
        agent=MAIN_AGENT_MODEL_SCOPE,
        stage=SYNTHESIS_STAGE,
    )


@traceable(name="Synthesis Node")
def run_synthesis(state: AgentState | MainState) -> AgentState | MainState:
    iteration_trace = _resolve_iteration_trace(state)
    if isinstance(state, MainState):
        child_failures = []
        for agent_state in state.agent_states:
            if agent_state.iteration_trace:
                continue
            if not agent_state.result.answer:
                continue
            child_failures.extend(
                entry
                for entry in agent_state.result.answer
                if isinstance(entry, str) and entry.strip()
            )
        if child_failures:
            state.result = AgentResult(
                answer=child_failures,
            )
            return state
    if not iteration_trace and not getattr(state, "goal_reached", False):
        state.result = AgentResult(answer=[])
        if isinstance(state, AgentState):
            state.goal_reached = True
        return state

    relevant_evidence_ids = _resolve_relevant_evidence_ids(state)
    evidence_bundle = build_evidence_bundle(iteration_trace)
    all_evidence_steps = build_evidence_steps(
        iteration_trace,
        evidence_bundle.evidence_views_by_step_id,
    )
    evidence_steps = filter_evidence_steps(all_evidence_steps, relevant_evidence_ids)
    if not evidence_steps:
        evidence_steps = all_evidence_steps

    prompt = build_synthesis_prompt(evidence=evidence_steps, state=state)
    prompt_text = prompt.prompt_text()
    prompt_input_object = prompt.to_log_input_object()
    llm = _build_synthesis_llm(state)
    started_at = perf_counter()
    response = llm.invoke(prompt_text)
    latency_ms = int((perf_counter() - started_at) * 1000)
    llm_call = record_llm_call(
        raw_response=response,
        model_name=_resolve_synthesis_model_name(state),
        conversation_id=state.conversation_id,
        roundtrip_id=state.roundtrip_id,
        user_id=state.user_profile.user_id,
        agent=MAIN_AGENT_MODEL_SCOPE,
        stage=SYNTHESIS_STAGE,
        callsite="shared_synthesis.run_synthesis",
        latency_ms=latency_ms,
        owner_agent_name=_resolve_agent_name(state),
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
        if isinstance(state, AgentState):
            state.goal_reached = True
        return state

    had_tool_results = any(bool(iteration.results) for iteration in iteration_trace)
    tool_summary = synthesis_result.tool_summary.model_dump() if had_tool_results else {}
    if had_tool_results:
        tool_summary["used_tools"] = _resolve_tool_summary_used_tools(state)
        tool_summary["freshness"] = _resolve_tool_summary_freshness(state)
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

    log_data = {
        "answer_preview": [block.content for block in synthesis_result.result[:3]],
        "next_question": synthesis_result.next_question,
        "relevant_evidence_ids": used_evidence_ids,
        "llm_usage": None if llm_call is None else serialize_llm_call_record(llm_call),
    }
    if isinstance(state, MainState):
        create_conversation_event(
            conversation_id=state.conversation_id,
            roundtrip_id=state.roundtrip_id,
            event_type=SYNTHESIS_KIND,
            source=_resolve_agent_name(state),
            agent_name=_resolve_agent_name(state),
            payload={
                "agent_name": _resolve_agent_name(state),
                "kind": SYNTHESIS_KIND,
                "data": log_data,
            },
        )
    else:
        create_conversation_event(
            conversation_id=state.conversation_id,
            roundtrip_id=state.roundtrip_id,
            event_type=SYNTHESIS_KIND,
            source=state.agent_profile.name,
            agent_name=state.agent_profile.name,
            payload={
                "agent_name": state.agent_profile.name,
                "kind": SYNTHESIS_KIND,
                "data": log_data,
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
    if isinstance(state, AgentState):
        state.goal_reached = True

    if state.roundtrip_id:
        log_roundtrip_prompt(
            roundtrip_id=state.roundtrip_id,
            agent=_resolve_agent_name(state),
            prompt_step=SYNTHESIS_PROMPT_STEP,
            prompt=prompt_text,
        )

    return state

