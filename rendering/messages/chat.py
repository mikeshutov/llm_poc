from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

import streamlit as st

from common.data import sanitize_for_json_storage
from conversation.conversation import generate_conversation_title
from conversation.models.conversation_models import ConversationRoundtrip
from conversation.repository.repo_factory import get_conversation_repo
from conversation.summary_service import rebuild_conversation_summaries
from agent_runtime.models.orchestrator_result import OrchestratorResult
from agent_runtime.models.orchestrator_payload import OrchestratorPayload
from rendering.feedback import render_feedback_controls
from rendering.rendering import (
    _format_roundtrip_duration,
    _format_roundtrip_usage_summary,
    fetch_llm_usage_for_roundtrip,
    format_timestamp,
    render_assistant_content,
    serialize_roundtrip_payload,
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
    payload, _ = answer.to_persistence_models()
    return sanitize_for_json_storage(payload.model_dump(exclude_none=True))


def ensure_messages_loaded(conversation_repository, conversation_id: str, limit: int = MESSAGE_HISTORY_LIMIT) -> None:
    if "messages" not in st.session_state or st.session_state.get("loaded_cid") != conversation_id:
        roundtrips = conversation_repository.list_roundtrips(
            UUID(conversation_id),
            limit=limit,
            newest_first=True,
        )
        st.session_state.messages = []
        for rt in roundtrips:
            ts = rt.created_at
            feedback = conversation_repository.get_roundtrip_feedback(rt.id)
            visible_messages = conversation_repository.list_messages(rt.id, include_internal=False)
            for message in visible_messages:
                if message.kind not in {"user", "assistant"}:
                    continue
                payload = (
                    serialize_roundtrip_payload(OrchestratorPayload.model_validate(message.payload))
                    if message.metadata.get("stage") == "synthesis" else None
                )
                st.session_state.messages.append(
                    {
                        ROLE_KEY: ROLE_USER if message.kind == "user" else ROLE_ASSISTANT,
                        CONTENT_KEY: message.content,
                        **({"payload": payload} if payload is not None else {}),
                        "assistant_follow_up": payload.get("next_question", "") if payload is not None else "",
                        "timestamp": message.created_at or ts,
                        "roundtrip_id": str(rt.id),
                        "model": message.metadata.get("model"),
                        "feedback_id": str(feedback.id) if feedback is not None else None,
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
    rendered_response = str(answer.raw_response or "")
    model = str(roundtrip.metadata.get("model", ""))

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
        "model": model,
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
            model=model,
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
