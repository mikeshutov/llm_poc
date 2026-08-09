import json

import streamlit as st

from common.message_constants import CONTENT_KEY, ROLE_DEBUG, ROLE_KEY


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
                st.json(content)
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


def _render_llm_usage_section(entry: dict) -> str | None:
    data = entry.get("data") or {}
    llm_usage = data.get("llm_usage")
    if not llm_usage:
        return None
    return f"LLM Usage:\n```json\n{_serialize_value(llm_usage)}\n```"


def _render_request_analysis_entry(entry: dict) -> list[str]:
    data = entry.get("data") or {}
    categories = data.get("applicable_tool_categories") or []
    requested_types = data.get("requested_user_attribute_types") or []
    confidence = data.get("context_answer_confidence")
    goal = data.get("goal") or ""

    parts = ["**Request Analysis**"]
    if confidence is not None:
        if confidence >= 0.8:
            summary = f"Can answer directly (context answer confidence: {confidence:.0%})"
        elif categories:
            summary = f"Categories: {', '.join(categories)} (context answer confidence: {confidence:.0%})"
        else:
            summary = f"No matching categories, using all tools (context answer confidence: {confidence:.0%})"
    elif categories:
        summary = f"Categories: {', '.join(categories)}"
    else:
        summary = "No matching categories, using all tools"
    parts.append(summary)
    if goal:
        parts.append(f"Goal: {goal}")
    if requested_types:
        parts.append(f"Useful stored attributes: {', '.join(requested_types)}")
    else:
        parts.append("Useful stored attributes: none requested")
    llm_usage = _render_llm_usage_section(entry)
    if llm_usage:
        parts.append(llm_usage)
    return parts


def _render_profile_load_entry(entry: dict) -> list[str]:
    data = entry.get("data") or {}
    requested_types = data.get("requested_user_attribute_types") or []
    loaded_attribute_types = data.get("loaded_attribute_types") or []
    loaded_attributes = data.get("loaded_attributes") or []
    loaded_count = data.get("loaded_attribute_count", 0)

    parts = ["**User Profile Loaded**"]
    if requested_types:
        parts.append(f"Requested attributes: {', '.join(requested_types)}")
    else:
        parts.append("Requested attributes: none")

    if loaded_attribute_types:
        parts.append(f"Loaded attribute types: {', '.join(loaded_attribute_types)}")
    else:
        parts.append("Loaded attribute types: none")

    parts.append(f"Loaded profile attributes: {loaded_count}")

    if loaded_attributes:
        lines = []
        for attribute in loaded_attributes:
            attribute_type = attribute.get("attribute_type") or "unknown"
            values = attribute.get("value") or []
            group_key = attribute.get("group_key")
            suffix = f" [{group_key}]" if group_key else ""
            value_text = ", ".join(str(value) for value in values)
            lines.append(f"- {attribute_type}{suffix}: {value_text}")
        parts.append("\n".join(lines))
    return parts


def _render_plan_entry(entry: dict) -> list[str]:
    data = entry.get("data") or {}
    step_plans = data.get("step_plans") or []
    final_answer = data.get("final_answer")

    parts = ["**Plan Generated**"]
    if step_plans:
        parts.append("\n".join(f"{index}. {step_plan}" for index, step_plan in enumerate(step_plans, start=1)))
    elif final_answer:
        parts.append("Answer directly without tool calls.")
    else:
        parts.append("No steps were generated.")
    llm_usage = _render_llm_usage_section(entry)
    if llm_usage:
        parts.append(llm_usage)
    return parts


def _render_synthesis_entry(entry: dict) -> list[str]:
    data = entry.get("data") or {}
    answer_preview = data.get("answer_preview") or []
    follow_up = data.get("follow_up") or ""
    clarifying_question = data.get("clarifying_question") or ""

    parts = ["**Synthesis**"]
    if answer_preview:
        parts.append("Answer preview:\n" + "\n".join(f"- {item}" for item in answer_preview))
    if follow_up:
        parts.append(f"Follow up: {follow_up}")
    if clarifying_question:
        parts.append(f"Clarifying question: {clarifying_question}")
    llm_usage = _render_llm_usage_section(entry)
    if llm_usage:
        parts.append(llm_usage)
    return parts


def _render_evaluator_entry(entry: dict) -> list[str]:
    data = entry.get("data") or {}
    parts = ["**Evaluation**"]
    parts.append(f"Satisfied: {bool(data.get('satisfied'))}")
    relevant_evidence = data.get("relevant_evidence") or []
    if relevant_evidence:
        parts.append("Relevant evidence: " + ", ".join(str(item) for item in relevant_evidence))
    missing_information = data.get("missing_information") or []
    if missing_information:
        parts.append("Missing information:\n" + "\n".join(f"- {item}" for item in missing_information))
    refined_goal = (data.get("refined_goal") or "").strip()
    if refined_goal:
        parts.append(f"Refined goal: {refined_goal}")
    parse_error = data.get("parse_error") or ""
    if parse_error:
        parts.append(f"Parse error: {parse_error}")
    llm_usage = _render_llm_usage_section(entry)
    if llm_usage:
        parts.append(llm_usage)
    return parts
def _render_tool_call_entry(entry: dict) -> list[str]:
    parts = ["**Tool Call**"]
    data = entry.get("data") or {}
    step_plan = data.get("step_plan") or entry.get("summary")
    if step_plan:
        parts.append(f"Working on: {step_plan}")

    tool_name = entry.get("tool_name") or data.get("tool_name")
    step_id = entry.get("step_id") or data.get("step_id")
    iteration = entry.get("iteration")
    request = entry.get("request")
    response = entry.get("response")
    error = entry.get("error")
    metadata = entry.get("metadata") or {}

    if tool_name:
        parts.append(f"Tool: `{tool_name}`")
    if step_id:
        parts.append(f"Step: `{step_id}`")
    if iteration is not None:
        parts.append(f"Iteration: `{iteration}`")
    if request is not None:
        parts.append(f"Request:\n```json\n{_serialize_value(request)}\n```")
    if response is not None:
        parts.append(f"Response:\n```json\n{_serialize_value(response)}\n```")
    if error:
        parts.append(f"Error:\n```text\n{error}\n```")
    if metadata:
        parts.append(f"Metadata:\n```json\n{_serialize_value(metadata)}\n```")
    return parts


def _assemble_log_message(entry: dict) -> str:
    kind = entry.get("kind") or "event"

    if kind == REQUEST_ANALYSIS_KIND:
        parts = _render_request_analysis_entry(entry)
    elif kind == PROFILE_LOAD_KIND:
        parts = _render_profile_load_entry(entry)
    elif kind == PLAN_KIND:
        parts = _render_plan_entry(entry)
    elif kind == EVALUATOR_KIND:
        parts = _render_evaluator_entry(entry)
    elif kind == SYNTHESIS_KIND:
        parts = _render_synthesis_entry(entry)
    elif kind == TOOL_CALL_KIND:
        parts = _render_tool_call_entry(entry)
    else:
        parts = [f"**{entry.get('title', 'Log Entry')}**"]
        for key in ("summary", "details"):
            value = entry.get(key)
            if value:
                parts.append(value)

    return "\n\n".join(part for part in parts if part)


def render_agent_logs(agent_logs: dict[str, list[dict]] | None) -> None:
    if not agent_logs:
        return

    for agent_name, entries in agent_logs.items():
        if not entries:
            continue
        with st.expander(f"{agent_name} log"):
            for index, entry in enumerate(entries):
                st.code(_assemble_log_message(entry), language="markdown")
                if index < len(entries) - 1:
                    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
