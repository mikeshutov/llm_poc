from datetime import datetime, timezone
from uuid import UUID

import streamlit as st

from conversation.conversation import generate_conversation_summary, generate_conversation_title
from conversation.models.conversation_models import ConversationRoundtrip
from conversation.repository.repo_factory import get_conversation_repo
from rendering.rendering import render_assistant_content, _format_timestamp
from common.message_constants import CONTENT_KEY, ROLE_ASSISTANT, ROLE_KEY, ROLE_USER


def ensure_messages_loaded(conversation_repository, conversation_id: str, limit: int = 10) -> None:
    if "messages" not in st.session_state or st.session_state.get("loaded_cid") != conversation_id:
        roundtrips = conversation_repository.list_roundtrips(
            UUID(conversation_id),
            limit=limit,
        )
        st.session_state.messages = []
        for rt in roundtrips:
            ts = rt.created_at if hasattr(rt, "created_at") else None
            st.session_state.messages.append({ROLE_KEY: ROLE_USER, CONTENT_KEY: rt.user_prompt, "timestamp": ts})
            payload = rt.response_payload if isinstance(rt.response_payload, dict) else None
            st.session_state.messages.append(
                {
                    ROLE_KEY: ROLE_ASSISTANT,
                    CONTENT_KEY: rt.generated_response,
                    "payload": payload,
                    "timestamp": ts,
                }
            )
        st.session_state.loaded_cid = conversation_id


def render_messages(conversation_repository, conversation_id: str, render_message, limit: int = 10) -> None:
    ensure_messages_loaded(conversation_repository, conversation_id, limit=limit)
    for msg in st.session_state.messages:
        render_message(msg)


def append_assistant_response(
    conversation_id: str,
    user_query: str,
    answer,
    roundtrip: ConversationRoundtrip,
    summary_every: int = 5,
) -> None:
    conversation_repository = get_conversation_repo()

    payload = {
        "response": answer.raw_response,
        "cards": answer.cards,
        "follow_up": answer.follow_up,
        "clarifying_question": answer.clarifying_question,
    }

    now = datetime.now(timezone.utc)
    st.session_state.messages.append(
        {
            ROLE_KEY: ROLE_ASSISTANT,
            CONTENT_KEY: answer.raw_response,
            "payload": payload,
            "timestamp": now,
        }
    )
    with st.chat_message(ROLE_ASSISTANT):
        render_assistant_content(answer.raw_response, payload, _format_timestamp(now))

    if summary_every > 0 and (roundtrip.message_index + 1) % summary_every == 0:
        latest_summary = conversation_repository.get_latest_summary(UUID(conversation_id))
        last_cutoff = latest_summary.message_index_cutoff if latest_summary else -1
        new_roundtrips = conversation_repository.list_roundtrips(
            UUID(conversation_id),
            limit=summary_every,
            after_message_index=last_cutoff,
        )
        summary_text = generate_conversation_summary(
            latest_summary.summary if latest_summary else None,
            new_roundtrips,
        )
        conversation_repository.create_summary(
            UUID(conversation_id),
            summary_text,
            message_index_cutoff=roundtrip.message_index,
        )
    if roundtrip.message_index == 0:
        conversation_repository.set_conversation_title(
            conversation_id,
            generate_conversation_title(user_query),
        )
        st.rerun()
