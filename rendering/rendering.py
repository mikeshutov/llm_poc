from datetime import datetime, timezone

import streamlit as st

from rendering.debug import debug_render_message
from rendering.cards import render_cards
from common.message_constants import CONTENT_KEY, ROLE_ASSISTANT, ROLE_DEBUG, ROLE_KEY


def _format_timestamp(ts) -> str | None:
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
        return f"{hour}:{local.strftime('%M %p')} · {local.strftime('%b')} {day}, {local.strftime('%Y')}"
    return None


def render_assistant_content(content: str, payload: dict | None, timestamp: str | None = None) -> None:
    cards = None
    follow_up = None
    if isinstance(payload, dict):
        cards = payload.get("cards")
        if cards is None:
            cards = payload.get("products")
        follow_up = payload.get("follow_up")
    has_cards = isinstance(cards, list) and bool(cards)
    has_follow_up = isinstance(follow_up, str) and bool(follow_up)

    if has_follow_up and not has_cards:
        st.markdown(f"{content}\n\n{follow_up}")
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

    if has_follow_up and has_cards:
        st.markdown(follow_up)

    if timestamp:
        st.caption(timestamp)


def render_message(msg: dict) -> None:
    role = msg[ROLE_KEY]
    content = msg[CONTENT_KEY]
    content_title = msg.get("title", "Debug")
    timestamp = _format_timestamp(msg.get("timestamp"))
    if msg.get("status"):
        with st.chat_message("assistant", avatar=":material/more_horiz:"):
            st.markdown(content)
    elif role == ROLE_DEBUG:
        debug_render_message(content, content_title)
    else:
        with st.chat_message(role):
            if role == ROLE_ASSISTANT:
                render_assistant_content(content, msg.get("payload"), timestamp)
            else:
                st.write(content)
                if timestamp:
                    st.caption(timestamp)
