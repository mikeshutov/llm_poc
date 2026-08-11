import json

import streamlit as st

from common.message_constants import CONTENT_KEY, ROLE_DEBUG, ROLE_KEY
from request_orchestrator.models.evaluation_result import EVALUATION_STATUS_RETRYABLE
from request_orchestrator.models.evaluation_result import EVALUATION_STATUS_RETRYABLE


REQUEST_ANALYSIS_KIND = "request_analysis"
PROFILE_LOAD_KIND = "profile_load"
PLAN_KIND = "plan"
EVALUATOR_KIND = "evaluator"
TOOL_CALL_KIND = "tool_call"
SYNTHESIS_KIND = "synthesis"


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


def _serialize_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, sort_keys=True, default=str)


def _build_request_analysis_payload(entry: dict) -> dict:
    data = entry.get("data") or {}
    return {
        "title": "Request Analysis",
        "categories": data.get("applicable_tool_categories") or [],
        "goal": data.get("goal") or "",
        "requested_user_attribute_types": data.get("requested_user_attribute_types") or [],
        "llm_usage": data.get("llm_usage"),
    }


def _build_profile_load_payload(entry: dict) -> dict:
    data = entry.get("data") or {}
    return {
        "title": "User Profile Loaded",
        "requested_user_attribute_types": data.get("requested_user_attribute_types") or [],
        "loaded_attribute_types": data.get("loaded_attribute_types") or [],
        "loaded_attribute_count": data.get("loaded_attribute_count", 0),
        "loaded_attributes": data.get("loaded_attributes") or [],
    }


def _build_plan_payload(entry: dict) -> dict:
    data = entry.get("data") or {}
    return {
        "title": "Plan Generated",
        "status": entry.get("status") or data.get("planner_status") or "",
        "reason": data.get("planner_reason") or "",
        "step_plans": data.get("step_plans") or [],
        "llm_usage": data.get("llm_usage"),
    }


def _build_synthesis_payload(entry: dict) -> dict:
    data = entry.get("data") or {}
    return {
        "title": "Synthesis",
        "answer_preview": data.get("answer_preview") or [],
        "follow_up": data.get("follow_up") or "",
        "clarifying_question": data.get("clarifying_question") or "",
        "relevant_evidence_ids": data.get("relevant_evidence_ids") or [],
        "llm_usage": data.get("llm_usage"),
    }


def _build_evaluator_payload(entry: dict) -> dict:
    data = entry.get("data") or {}
    return {
        "title": "Evaluation",
        "status": data.get("status") or entry.get("status") or EVALUATION_STATUS_RETRYABLE,
        "relevant_evidence": data.get("relevant_evidence") or [],
        "missing_information": data.get("missing_information") or [],
        "refined_goal": (data.get("refined_goal") or "").strip(),
        "parse_error": data.get("parse_error") or "",
        "llm_usage": data.get("llm_usage"),
    }


def _build_tool_call_payload(entry: dict) -> dict:
    data = entry.get("data") or {}
    return {
        "title": "Tool Call",
        "step_plan": data.get("step_plan") or entry.get("summary") or "",
        "tool_name": entry.get("tool_name") or data.get("tool_name") or "",
        "step_id": entry.get("step_id") or data.get("step_id") or "",
        "iteration": entry.get("iteration"),
        "request": entry.get("request"),
        "response": entry.get("response"),
        "error": entry.get("error") or "",
        "latency_ms": data.get("latency_ms"),
        "metadata": entry.get("metadata") or {},
    }


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
    else:
        payload = dict(entry)
        payload.setdefault("title", entry.get("title", "Log Entry"))

    title = str(payload.get("title") or entry.get("title") or "Log Entry")
    return title, payload


def render_agent_logs(agent_logs: dict[str, list[dict]] | None) -> None:
    if not agent_logs:
        return

    for agent_name, entries in agent_logs.items():
        if not entries:
            continue
        with st.expander(f"{agent_name} log"):
            for index, entry in enumerate(entries):
                title, payload = _build_log_payload(entry)
                st.markdown(f"**{title}**")
                st.json(payload, expanded=False)
                if index < len(entries) - 1:
                    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
