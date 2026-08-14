import streamlit as st
from pydantic import BaseModel, Field

from common.config import CONTENT_KEY, ROLE_DEBUG, ROLE_KEY
from request_orchestrator.models.evaluation_result import EVALUATION_STATUS_RETRYABLE


REQUEST_ANALYSIS_KIND = "request_analysis"
PROFILE_LOAD_KIND = "profile_load"
PLAN_KIND = "plan"
EVALUATOR_KIND = "evaluator"
TOOL_CALL_KIND = "tool_call"
SYNTHESIS_KIND = "synthesis"
LLM_CALL_KIND = "llm_call"
ORCHESTRATOR_AGENT_NAME = "request_orchestrator"
DEFAULT_AGENT_LOG_ORDER = [
    ORCHESTRATOR_AGENT_NAME,
    "profile_management",
    "main_agent",
]


class RequestAnalysisLogPayload(BaseModel):
    title: str = "Request Analysis"
    categories: list[str] = Field(default_factory=list)
    goal: str = ""
    requested_user_attribute_types: list[str] = Field(default_factory=list)


class ProfileLoadLogPayload(BaseModel):
    title: str = "User Profile Loaded"
    requested_user_attribute_types: list[str] = Field(default_factory=list)
    loaded_attribute_types: list[str] = Field(default_factory=list)
    loaded_attribute_count: int = 0
    loaded_attributes: list[dict] = Field(default_factory=list)


class PlanLogPayload(BaseModel):
    title: str = "Plan Generated"
    status: str = ""
    reason: str = ""
    step_plans: list[str] = Field(default_factory=list)


class SynthesisLogPayload(BaseModel):
    title: str = "Synthesis"
    answer_preview: list[str] = Field(default_factory=list)
    next_question: str = ""
    relevant_evidence_ids: list[str] = Field(default_factory=list)


class EvaluatorLogPayload(BaseModel):
    title: str = "Evaluation"
    status: str = EVALUATION_STATUS_RETRYABLE
    relevant_evidence: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    refined_goal: str = ""
    parse_error: str = ""


class ToolCallLogPayload(BaseModel):
    title: str = "Tool Call"
    step_plan: str = ""
    tool_name: str = ""
    step_id: str = ""
    iteration: int | None = None
    request: object | None = None
    response: object | None = None
    error: str = ""
    latency_ms: int | None = None
    metadata: dict = Field(default_factory=dict)


class LlmCallLogPayload(BaseModel):
    title: str = "LLM Call"
    model_scope: str = ""
    model: str = ""
    stage: str = ""
    callsite: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0
    computed_input_cost: str = ""
    computed_output_cost: str = ""
    computed_total_cost: str = ""
    latency_ms: int | None = None
    metadata: dict = Field(default_factory=dict)


def debug_render_message(content, content_title: str) -> None:
    with st.chat_message("assistant", avatar=":material/edit:"):
        with st.expander(content_title):
            if isinstance(content, (dict, list)):
                st.json(content, expanded=False)
            else:
                st.code(str(content), language="text")


def emit_debug_message(content, content_title: str) -> None:
    try:
        if "messages" not in st.session_state:
            st.session_state.messages = []
        st.session_state.messages.append(
            {
                ROLE_KEY: ROLE_DEBUG,
                CONTENT_KEY: content,
                "title": content_title,
            }
        )
        debug_render_message(content, content_title)
    except Exception:
        pass

def _build_request_analysis_payload(entry: dict) -> dict:
    data = entry.get("data") or {}
    return RequestAnalysisLogPayload(
        categories=data.get("applicable_tool_categories") or [],
        goal=data.get("goal") or "",
        requested_user_attribute_types=data.get("requested_user_attribute_types") or [],
    ).model_dump()


def _build_profile_load_payload(entry: dict) -> dict:
    data = entry.get("data") or {}
    return ProfileLoadLogPayload(
        requested_user_attribute_types=data.get("requested_user_attribute_types") or [],
        loaded_attribute_types=data.get("loaded_attribute_types") or [],
        loaded_attribute_count=data.get("loaded_attribute_count", 0),
        loaded_attributes=data.get("loaded_attributes") or [],
    ).model_dump()


def _build_plan_payload(entry: dict) -> dict:
    data = entry.get("data") or {}
    return PlanLogPayload(
        status=entry.get("status") or data.get("planner_status") or "",
        reason=data.get("planner_reason") or "",
        step_plans=data.get("step_plans") or [],
    ).model_dump()


def _build_synthesis_payload(entry: dict) -> dict:
    data = entry.get("data") or {}
    return SynthesisLogPayload(
        answer_preview=data.get("answer_preview") or [],
        next_question=data.get("next_question") or "",
        relevant_evidence_ids=data.get("relevant_evidence_ids") or [],
    ).model_dump()


def _build_evaluator_payload(entry: dict) -> dict:
    data = entry.get("data") or {}
    return EvaluatorLogPayload(
        status=data.get("status") or entry.get("status") or EVALUATION_STATUS_RETRYABLE,
        relevant_evidence=data.get("relevant_evidence") or [],
        missing_information=data.get("missing_information") or [],
        refined_goal=(data.get("refined_goal") or "").strip(),
        parse_error=data.get("parse_error") or "",
    ).model_dump()


def _build_tool_call_payload(entry: dict) -> dict:
    data = entry.get("data") or {}
    return ToolCallLogPayload(
        step_plan=data.get("step_plan") or entry.get("summary") or "",
        tool_name=entry.get("tool_name") or data.get("tool_name") or "",
        step_id=entry.get("step_id") or data.get("step_id") or "",
        iteration=entry.get("iteration"),
        request=entry.get("request"),
        response=entry.get("response"),
        error=entry.get("error") or "",
        latency_ms=data.get("latency_ms"),
        metadata=entry.get("metadata") or {},
    ).model_dump()


def _build_llm_call_payload(entry: dict) -> dict:
    return LlmCallLogPayload(
        model_scope=entry.get("model_scope") or entry.get("agent") or "",
        model=entry.get("model") or "",
        stage=entry.get("stage") or "",
        callsite=entry.get("callsite") or "",
        input_tokens=entry.get("input_tokens") or 0,
        output_tokens=entry.get("output_tokens") or 0,
        total_tokens=entry.get("total_tokens") or 0,
        cached_input_tokens=entry.get("cached_input_tokens") or 0,
        computed_input_cost=str(entry.get("computed_input_cost") or ""),
        computed_output_cost=str(entry.get("computed_output_cost") or ""),
        computed_total_cost=str(entry.get("computed_total_cost") or ""),
        latency_ms=entry.get("latency_ms"),
        metadata=entry.get("metadata") or {},
    ).model_dump()


def _build_log_payload(entry: dict) -> tuple[str, dict]:
    kind = entry.get("kind") or "event"

    if kind == REQUEST_ANALYSIS_KIND:
        payload = _build_request_analysis_payload(entry)
    elif kind == PROFILE_LOAD_KIND:
        payload = _build_profile_load_payload(entry)
    elif kind == PLAN_KIND:
        payload = _build_plan_payload(entry)
    elif kind == EVALUATOR_KIND:
        payload = _build_evaluator_payload(entry)
    elif kind == SYNTHESIS_KIND:
        payload = _build_synthesis_payload(entry)
    elif kind == TOOL_CALL_KIND:
        payload = _build_tool_call_payload(entry)
    elif kind == LLM_CALL_KIND:
        payload = _build_llm_call_payload(entry)
    else:
        payload = dict(entry)
        payload.setdefault("title", entry.get("title", "Log Entry"))

    title = str(payload.get("title") or entry.get("title") or "Log Entry")
    return title, payload


def _ordered_agent_log_sections(agent_logs: dict[str, list[dict]] | None) -> list[tuple[str, list[dict]]]:
    if not agent_logs:
        return []

    ordered_names: list[str] = []
    for agent_name in DEFAULT_AGENT_LOG_ORDER:
        entries = agent_logs.get(agent_name)
        if entries:
            ordered_names.append(agent_name)

    remaining_names = sorted(
        agent_name
        for agent_name, entries in agent_logs.items()
        if entries and agent_name not in ordered_names
    )
    ordered_names.extend(remaining_names)
    return [(agent_name, agent_logs[agent_name]) for agent_name in ordered_names]


def _render_log_entries(entries: list[dict]) -> None:
    for index, entry in enumerate(entries):
        title, payload = _build_log_payload(entry)
        st.markdown(f"**{title}**")
        st.json(payload, expanded=False)
        if index < len(entries) - 1:
            st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)


def _split_orchestrator_entries_for_agents(entries: list[dict]) -> tuple[list[dict], list[dict]]:
    for index, entry in enumerate(entries):
        if entry.get("kind") == SYNTHESIS_KIND:
            return entries[:index], entries[index:]
    return entries, []


def render_agent_logs(agent_logs: dict[str, list[dict]] | None) -> None:
    ordered_sections = _ordered_agent_log_sections(agent_logs)
    if not ordered_sections:
        return

    top_level_name, top_level_entries = ordered_sections[0]
    if top_level_name == ORCHESTRATOR_AGENT_NAME:
        with st.expander(f"{top_level_name} log"):
            before_agent_sections, after_agent_sections = _split_orchestrator_entries_for_agents(top_level_entries)
            if before_agent_sections:
                _render_log_entries(before_agent_sections)
            for agent_name, entries in ordered_sections[1:]:
                if not entries:
                    continue
                with st.expander(f"{agent_name} log"):
                    _render_log_entries(entries)
            if after_agent_sections:
                _render_log_entries(after_agent_sections)
        return

    for agent_name, entries in ordered_sections:
        if not entries:
            continue
        with st.expander(f"{agent_name} log"):
            _render_log_entries(entries)
