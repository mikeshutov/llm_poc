import os
from datetime import datetime

import streamlit as st

from rendering.debug import debug_render_message, render_agent_logs
from rendering.cards import render_cards
from rendering.feedback import render_feedback_controls
from rendering.replay import render_replay_control
from common.message_constants import CONTENT_KEY, ROLE_ASSISTANT, ROLE_DEBUG, ROLE_KEY
from common.file_constants import FILES_DIR, IMAGE_MIME_PREFIX


def format_timestamp(ts) -> str | None:
    if ts is None:
        return None
    if isinstance(ts, str):
        try:
            ts = datetime.fromisoformat(ts)
        except ValueError:
            return ts
    if isinstance(ts, datetime):
        local = ts.astimezone()
        hour = local.strftime("%I").lstrip("0") or "12"
        day = local.strftime("%d").lstrip("0") or "1"
        return f"{hour}:{local.strftime('%M %p')} | {local.strftime('%b')} {day}, {local.strftime('%Y')}"
    return None


def _format_roundtrip_usage_summary(llm_usage: dict | None) -> str | None:
    if not isinstance(llm_usage, dict):
        return None
    summary = llm_usage.get("summary")
    if not isinstance(summary, dict):
        return None

    total_tokens = summary.get("total_tokens")
    total_cost = summary.get("computed_total_cost")
    if total_tokens is None and total_cost is None:
        return None

    parts: list[str] = []
    if total_tokens is not None:
        parts.append(f"Total Tokens: {int(total_tokens):,}")
    if total_cost is not None:
        parts.append(f"Estimated Cost: ${total_cost}")
    return " | ".join(parts) if parts else None


def _render_roundtrip_llm_usage(llm_usage: dict | None) -> None:
    if not isinstance(llm_usage, dict):
        return
    summary = llm_usage.get("summary") if isinstance(llm_usage.get("summary"), dict) else {}
    calls = llm_usage.get("calls") if isinstance(llm_usage.get("calls"), list) else []
    if not summary and not calls:
        return

    title = f"LLM Usage ({llm_usage.get('retrieved_call_count', len(calls))} calls)"
    with st.expander(title):
        if summary:
            st.json(summary)
        if calls:
            st.json(calls)


def render_assistant_content(content: str, payload: dict | None) -> None:
    cards = None
    next_question = None
    agent_logs = None
    llm_usage = None
    if isinstance(payload, dict):
        cards = payload.get("cards")
        if cards is None:
            cards = payload.get("products")
        follow_up = payload.get("follow_up")
        clarifying_question = payload.get("clarifying_question")
        agent_logs = payload.get("agent_logs")
        llm_usage = payload.get("llm_usage")
        if isinstance(clarifying_question, str) and clarifying_question:
            next_question = clarifying_question
        elif isinstance(follow_up, str) and follow_up:
            next_question = follow_up
    has_cards = isinstance(cards, list) and bool(cards)
    has_next_question = isinstance(next_question, str) and bool(next_question)

    if has_next_question and not has_cards:
        st.markdown(f"{content}\n\n{next_question}")
    else:
        st.markdown(content)

    if has_cards:
        render_cards(
            cards,
            heading_key="name",
            description_key="description",
            image_key="image_url",
            link_key="url",
        )

    if has_next_question and has_cards:
        st.markdown(next_question)

    _render_roundtrip_llm_usage(llm_usage)
    render_agent_logs(agent_logs)


def _render_file_preview(attached_file: dict) -> None:
    name = attached_file.get("name", "")
    mime = attached_file.get("type", "")
    path = os.path.join(FILES_DIR, name)
    if mime.startswith(IMAGE_MIME_PREFIX) and os.path.exists(path):
        st.image(path, width=200)
    else:
        icon = "[PDF]" if "pdf" in mime else "[FILE]"
        st.markdown(
            f"<span style='background:#f0f2f6;border-radius:6px;padding:3px 10px;font-size:0.85em'>{icon} {name}</span>",
            unsafe_allow_html=True,
        )


def render_message(msg: dict) -> None:
    role = msg[ROLE_KEY]
    content = msg[CONTENT_KEY]
    content_title = msg.get("title", "Debug")
    timestamp = format_timestamp(msg.get("timestamp"))
    if msg.get("status"):
        with st.chat_message("assistant", avatar=":material/more_horiz:"):
            st.markdown(content)
    elif role == ROLE_DEBUG:
        debug_render_message(content, content_title)
    else:
        with st.chat_message(role):
            if role == ROLE_ASSISTANT:
                render_assistant_content(content, msg.get("payload"))
                render_feedback_controls(
                    roundtrip_id=msg.get("roundtrip_id"),
                    model=msg.get("model"),
                    feedback_id=msg.get("feedback_id"),
                    timestamp=timestamp,
                    usage_summary=_format_roundtrip_usage_summary(msg.get("payload", {}).get("llm_usage") if isinstance(msg.get("payload"), dict) else None),
                )
            else:
                st.write(content)
                attached_file = msg.get("attached_file")
                if attached_file:
                    _render_file_preview(attached_file)
                render_replay_control(
                    roundtrip_id=msg.get("roundtrip_id"),
                    timestamp=timestamp,
                )
