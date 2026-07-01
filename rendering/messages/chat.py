from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

import streamlit as st

from conversation.conversation import generate_conversation_summary, generate_conversation_title
from conversation.models.conversation_models import ConversationRoundtrip
from agent.models.agent_result import AgentResult
from conversation.repository.repo_factory import get_conversation_repo
from llm.clients.embeddings import embed_text
from rendering.feedback import render_feedback_controls
from rendering.rendering import render_assistant_content, format_timestamp
from common.message_constants import (
    CONTENT_KEY,
    ROLE_ASSISTANT,
    ROLE_KEY,
    ROLE_USER,
    SUMMARY_BATCH_SIZE,
    SUMMARY_TRIGGER_SIZE,
)
from tool.repository.tool_call_repository import ToolCallRepository


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
                    "roundtrip_id": str(rt.id),
                    "model": rt.model,
                    "feedback_id": str(rt.feedback_id) if rt.feedback_id is not None else None,
                }
            )
        st.session_state.loaded_cid = conversation_id


def render_messages(conversation_repository, conversation_id: str, render_message, limit: int = 10) -> None:
    ensure_messages_loaded(conversation_repository, conversation_id, limit=limit)
    for msg in st.session_state.messages:
        render_message(msg)


def _update_top_level_conversation_summary(
    conversation_id: str,
    all_roundtrips: list[ConversationRoundtrip],
    tool_calls_by_roundtrip,
) -> None:
    if not all_roundtrips:
        return

    conversation_repository = get_conversation_repo()
    top_level_summary = generate_conversation_summary(
        all_roundtrips,
        tool_call_map=tool_calls_by_roundtrip,
    )
    summary_text = top_level_summary.conversation_summary.strip()
    summary_embedding = embed_text(summary_text) if summary_text else None
    conversation_repository.update_conversation_summary(
        UUID(conversation_id),
        top_level_summary.conversation_summary,
        summary_embedding=summary_embedding,
    )


def _update_batched_conversation_summaries(conversation_id: str) -> None:
    conversation_repository = get_conversation_repo()
    latest_summary = conversation_repository.get_latest_summary(UUID(conversation_id))
    last_cutoff = latest_summary.message_index_cutoff if latest_summary else -1
    previous_batch_summary = latest_summary.summary if latest_summary else ""

    while True:
        unsummarized_roundtrips = conversation_repository.list_roundtrips(
            UUID(conversation_id),
            limit=SUMMARY_TRIGGER_SIZE,
            after_message_index=last_cutoff,
        )
        if len(unsummarized_roundtrips) < SUMMARY_TRIGGER_SIZE:
            return

        batch_roundtrips = unsummarized_roundtrips[:SUMMARY_BATCH_SIZE]
        roundtrip_ids = [rt.id for rt in batch_roundtrips]
        tool_calls_by_roundtrip = ToolCallRepository().get_tool_calls_by_roundtrips(roundtrip_ids)
        batch_summary = generate_conversation_summary(
            batch_roundtrips,
            tool_call_map=tool_calls_by_roundtrip,
            previous_summary=previous_batch_summary,
        )
        last_cutoff = batch_roundtrips[-1].message_index
        previous_batch_summary = batch_summary.conversation_summary
        conversation_repository.create_summary(
            UUID(conversation_id),
            batch_summary.conversation_summary,
            message_index_cutoff=last_cutoff,
            tool_summary=batch_summary.tool_summary,
        )


def _update_conversation_summary(conversation_id: str, roundtrip: ConversationRoundtrip) -> None:
    if roundtrip.message_index < 1:
        return

    conversation_repository = get_conversation_repo()
    all_roundtrips = conversation_repository.list_roundtrips(
        UUID(conversation_id),
        limit=roundtrip.message_index + 1,
    )
    if not all_roundtrips:
        return

    roundtrip_ids = [rt.id for rt in all_roundtrips]
    tool_calls_by_roundtrip = ToolCallRepository().get_tool_calls_by_roundtrips(roundtrip_ids)
    _update_top_level_conversation_summary(
        conversation_id,
        all_roundtrips,
        tool_calls_by_roundtrip,
    )
    _update_batched_conversation_summaries(conversation_id)


def append_assistant_response(
    conversation_id: str,
    user_query: str,
    answer: AgentResult,
    roundtrip: ConversationRoundtrip,
) -> None:
    conversation_repository = get_conversation_repo()

    payload = {
        "response": answer.raw_response,
        "cards": answer.cards,
        "follow_up": answer.follow_up,
        "clarifying_question": answer.clarifying_question,
    }

    now = datetime.now(timezone.utc)
    assistant_message = {
        ROLE_KEY: ROLE_ASSISTANT,
        CONTENT_KEY: answer.raw_response,
        "payload": payload,
        "timestamp": now,
        "roundtrip_id": str(roundtrip.id),
        "model": roundtrip.model,
        "feedback_id": None,
    }
    st.session_state.messages.append(assistant_message)
    with st.chat_message(ROLE_ASSISTANT):
        render_assistant_content(answer.raw_response, payload)
        render_feedback_controls(
            roundtrip_id=roundtrip.id,
            model=roundtrip.model,
            feedback_id=None,
            timestamp=format_timestamp(now),
        )

    _update_conversation_summary(conversation_id, roundtrip)

    if roundtrip.message_index == 0:
        conversation_repository.set_conversation_title(
            conversation_id,
            generate_conversation_title(user_query),
        )
        st.rerun()
