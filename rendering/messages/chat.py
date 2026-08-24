from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

import streamlit as st

from common.data import sanitize_for_json_storage
from conversation.conversation import generate_conversation_title
from conversation.models.conversation_models import ConversationRoundtrip
from conversation.repository.repo_factory import get_conversation_repo
from conversation.summary_service import rebuild_conversation_summaries
from request_orchestrator.models.orchestrator_result import OrchestratorResult
from rendering.feedback import render_feedback_controls
from rendering.rendering import (
    _format_roundtrip_duration,
    _format_roundtrip_usage_summary,
    fetch_llm_usage_for_roundtrip,
    format_timestamp,
    render_assistant_content,
)
from common.config import (
    CONTENT_KEY,
    ROLE_ASSISTANT,
    ROLE_KEY,
    ROLE_USER,
    SUMMARY_BATCH_SIZE,
    SUMMARY_TRIGGER_SIZE,
)


MESSAGE_HISTORY_LIMIT = 10


def _build_answer_payload(answer: OrchestratorResult) -> dict:
    return sanitize_for_json_storage(answer.to_payload_model().model_dump(exclude_none=True))

def ensure_messages_loaded(conversation_repository, conversation_id: str, limit: int = MESSAGE_HISTORY_LIMIT) -> None:
    if "messages" not in st.session_state or st.session_state.get("loaded_cid") != conversation_id:
        roundtrips = conversation_repository.list_roundtrips(
            UUID(conversation_id),
            limit=limit,
            newest_first=True,
        )
        st.session_state.messages = []
        for rt in roundtrips:
            ts = rt.created_at if hasattr(rt, "created_at") else None
            st.session_state.messages.append({ROLE_KEY: ROLE_USER, CONTENT_KEY: rt.user_prompt, "timestamp": ts, "roundtrip_id": str(rt.id)})
            payload = rt.response_payload if isinstance(rt.response_payload, dict) else None
            st.session_state.messages.append(
                {
                    ROLE_KEY: ROLE_ASSISTANT,
                    CONTENT_KEY: rt.generated_response,
                    "payload": payload,
                    "assistant_follow_up": rt.assistant_follow_up,
                    "timestamp": ts,
                    "roundtrip_id": str(rt.id),
                    "model": rt.model,
                    "feedback_id": str(rt.feedback_id) if rt.feedback_id is not None else None,
                }
            )
        st.session_state.loaded_cid = conversation_id


def render_messages(conversation_repository, conversation_id: str, render_message, limit: int = MESSAGE_HISTORY_LIMIT) -> None:
    ensure_messages_loaded(conversation_repository, conversation_id, limit=limit)
    for msg in st.session_state.messages:
        render_message(msg)



def _update_conversation_summary(conversation_id: str, roundtrip: ConversationRoundtrip) -> None:
    if roundtrip.message_index < 1:
        return

    rebuild_conversation_summaries(
        conversation_id,
        summary_batch_size=SUMMARY_BATCH_SIZE,
        summary_trigger_size=SUMMARY_TRIGGER_SIZE,
    )


def append_assistant_response(
    conversation_id: str,
    user_query: str,
    answer: OrchestratorResult,
    roundtrip: ConversationRoundtrip,
) -> None:
    conversation_repository = get_conversation_repo()

    payload = _build_answer_payload(answer)
    rendered_response = str(answer.raw_response or roundtrip.generated_response or "")

    now = datetime.now(timezone.utc)
    duration = _format_roundtrip_duration(payload)
    footer_timestamp = " | ".join(part for part in [duration, format_timestamp(now)] if part) or None
    assistant_message = {
        ROLE_KEY: ROLE_ASSISTANT,
        CONTENT_KEY: rendered_response,
        "payload": payload,
        "assistant_follow_up": answer.next_question,
        "timestamp": now,
        "roundtrip_id": str(roundtrip.id),
        "model": roundtrip.model,
        "feedback_id": None,
    }
    st.session_state.messages.append(assistant_message)
    with st.chat_message(ROLE_ASSISTANT):
        render_assistant_content(
            rendered_response,
            payload,
            roundtrip_id=str(roundtrip.id),
            assistant_follow_up=answer.next_question,
        )
        render_feedback_controls(
            roundtrip_id=roundtrip.id,
            model=roundtrip.model,
            sources_payload=payload,
            feedback_id=None,
            timestamp=footer_timestamp,
            usage_summary=_format_roundtrip_usage_summary(fetch_llm_usage_for_roundtrip(str(roundtrip.id))),
        )

    _update_conversation_summary(conversation_id, roundtrip)

    if roundtrip.message_index == 0:
        conversation_repository.set_conversation_title(
            conversation_id,
            generate_conversation_title(user_query),
        )
        st.rerun()
