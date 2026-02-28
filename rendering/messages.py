from uuid import UUID
from dataclasses import asdict

import streamlit as st

from agent.repository.repo_factory import get_tool_call_repo
from conversation.conversation import generate_conversation_title, generate_conversation_summary
from conversation.repository.repo_factory import get_conversation_repo
from rendering.rendering import render_assistant_content
from common.message_constants import CONTENT_KEY, ROLE_ASSISTANT, ROLE_KEY, ROLE_USER


def ensure_messages_loaded(conversation_repository, conversation_id: str, limit: int = 10) -> None:
    if "messages" not in st.session_state or st.session_state.get("loaded_cid") != conversation_id:
        roundtrips = conversation_repository.list_roundtrips(
            UUID(conversation_id),
            limit=limit,
        )
        st.session_state.messages = []
        for rt in roundtrips:
            st.session_state.messages.append({ROLE_KEY: ROLE_USER, CONTENT_KEY: rt.user_prompt})
            payload = rt.response_payload if isinstance(rt.response_payload, dict) else None
            st.session_state.messages.append(
                {
                    ROLE_KEY: ROLE_ASSISTANT,
                    CONTENT_KEY: rt.generated_response,
                    "payload": payload,
                }
            )
        st.session_state.loaded_cid = conversation_id


def render_messages(messages, render_message) -> None:
    for msg in messages:
        render_message(msg)


def append_assistant_response(
    conversation_id: str,
    user_query: str,
    answer,
    parsed_query: dict | None = None,
    tool_traces: list | None = None,
    summary_every: int = 5,
) -> None:
    conversation_repository = get_conversation_repo()
    tool_call_repository = get_tool_call_repo()

    payload = {
        "response": answer.response,
        "cards": answer.cards,
        "follow_up": answer.follow_up,
    }
    st.session_state.messages.append(
        {
            ROLE_KEY: ROLE_ASSISTANT,
            CONTENT_KEY: answer.response,
            "payload": payload,
        }
    )
    with st.chat_message(ROLE_ASSISTANT):
        render_assistant_content(answer.response, payload)

    append_result = conversation_repository.append_roundtrip(
        conversation_id,
        user_query,
        answer.response,
        response_payload=payload,
        parsed_query=parsed_query,
        metadata={},
    )
    if tool_traces:
        trace_dicts = [asdict(t) if hasattr(t, "__dataclass_fields__") else t for t in tool_traces]
        tool_call_repository.append_tool_calls(
            append_result.id,
            trace_dicts,
        )
            
    if summary_every > 0 and (append_result.message_index + 1) % summary_every == 0:
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
            message_index_cutoff=append_result.message_index,
        )
    if append_result.message_index == 0:
        conversation_repository.set_conversation_title(conversation_id, generate_conversation_title(user_query))
        st.rerun()
