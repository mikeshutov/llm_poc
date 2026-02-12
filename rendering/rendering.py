import streamlit as st

from rendering.cards import render_cards


def render_assistant_content(content: str, payload: dict | None) -> None:
    cards = None
    follow_up = None
    if isinstance(payload, dict):
        cards = payload.get("cards")
        if cards is None:
            cards = payload.get("products")
        follow_up = payload.get("follow_up")

    has_cards = isinstance(cards, list) and bool(cards)
    if isinstance(follow_up, str) and follow_up and not has_cards:
        st.info(f"{content}\n\n{follow_up}")
    else:
        st.info(content)

    if has_cards:
        render_cards(
            cards,
            heading_key="name",
            description_key="description",
            image_key="image_url",
            link_key="url",
        )

    if isinstance(follow_up, str) and follow_up and has_cards:
        st.info(follow_up)


def render_message(msg: dict) -> None:
    role = msg["role"]
    content = msg["content"]
    content_title = msg.get("title", "Debug")
    if role == "debug":
        debug_render_message(content, content_title)
    else:
        with st.chat_message(role):
            if role == "assistant":
                render_assistant_content(content, msg.get("payload"))
            else:
                st.write(content)


def debug_render_message(content, content_title: str) -> None:
    with st.chat_message("assistant", avatar="🧪"):
        st.markdown(content_title)
        with st.expander("Debug"):
            st.json(content)
