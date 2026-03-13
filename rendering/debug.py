import json

import streamlit as st

from common.message_constants import CONTENT_KEY, ROLE_ASSISTANT, ROLE_DEBUG, ROLE_KEY


def debug_render_message(content, content_title: str) -> None:
    with st.chat_message("assistant", avatar=":material/edit:"):
        st.markdown(content_title)
        with st.expander("Debug"):
            st.json(content)


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


def emit_status_message(content: str) -> None:
    try:
        if "messages" not in st.session_state:
            st.session_state.messages = []
        st.session_state.messages.append(
            {
                ROLE_KEY: ROLE_ASSISTANT,
                CONTENT_KEY: content,
                "status": True,
            }
        )
        with st.chat_message("assistant", avatar=":material/more_horiz:"):
            st.markdown(content)
    except Exception:
        pass


def build_classify_status_message(categories: list[str], can_answer_without_tools: bool = False, confidence: float = 0.0) -> str:
    if can_answer_without_tools:
        return f"**Classification:** can answer directly (confidence: {confidence:.0%})"
    if categories:
        return f"**Classified as:** {', '.join(categories)} (confidence: {confidence:.0%})"
    return "**Classification:** no matching categories, using all tools"


def build_plan_status_message(step_plans: list[str], final_answer: str | None = None) -> str:
    if step_plans:
        plan_text = "\n".join(
            f"{index}. {step_plan}"
            for index, step_plan in enumerate(step_plans, start=1)
        )
    elif final_answer:
        plan_text = "Answer directly without tool calls."
    else:
        plan_text = "No steps were generated."
    return f"**Plan generated**\n\n{plan_text}"


def build_step_status_message(step_plan: str, tool_name: str, args: dict) -> str:
    return (
        f"Working on: {step_plan}\n\n"
        f"Using `{tool_name}` with:\n"
        f"```json\n{json.dumps(args, indent=2, sort_keys=True)}\n```"
    )
