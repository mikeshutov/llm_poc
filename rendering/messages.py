from uuid import UUID

import streamlit as st

from rendering.rendering import render_assistant_content


def ensure_messages_loaded(conversation_repository, conversation_id: str, limit: int = 10) -> None:
    if "messages" not in st.session_state or st.session_state.get("loaded_cid") != conversation_id:
        roundtrips = conversation_repository.list_roundtrips(
            UUID(conversation_id),
            limit=limit,
        )
        st.session_state.messages = []
        for rt in roundtrips:
            st.session_state.messages.append({"role": "user", "content": rt.user_prompt})
            payload = rt.response_payload if isinstance(rt.response_payload, dict) else None
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": rt.generated_response,
                    "payload": payload,
                }
            )
        st.session_state.loaded_cid = conversation_id


def render_messages(messages, render_message) -> None:
    for msg in messages:
        render_message(msg)


def append_assistant_response(
    conversation_repository,
    conversation_id: str,
    user_query: str,
    answer,
    generate_title_fn,
    generate_summary_fn,
    parsed_query: dict | None = None,
    summary_every: int = 5,
) -> None:
    payload = {
        "response": answer.response,
        "cards": answer.cards,
        "follow_up": answer.follow_up,
    }
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer.response,
            "payload": payload,
        }
    )
    with st.chat_message("assistant"):
        render_assistant_content(answer.response, payload)

    append_result = conversation_repository.append_roundtrip(
        conversation_id,
        user_query,
        answer.response,
        response_payload=payload,
        parsed_query=parsed_query,
        metadata={},
    )
    if summary_every > 0 and (append_result.message_index + 1) % summary_every == 0:
        latest_summary = conversation_repository.get_latest_summary(UUID(conversation_id))
        last_cutoff = latest_summary.message_index_cutoff if latest_summary else -1
        new_roundtrips = conversation_repository.list_roundtrips(
            UUID(conversation_id),
            limit=summary_every,
            after_message_index=last_cutoff,
        )
        summary_text = generate_summary_fn(
            latest_summary.summary if latest_summary else None,
            new_roundtrips,
        )
        conversation_repository.create_summary(
            UUID(conversation_id),
            summary_text,
            message_index_cutoff=append_result.message_index,
        )
    if append_result.message_index == 0:
        generated_title = generate_title_fn(user_query)
        conversation_repository.set_conversation_title(conversation_id, generated_title)
        st.rerun()
