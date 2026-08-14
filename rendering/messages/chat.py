from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

import streamlit as st

from conversation.conversation import generate_conversation_title
from conversation.models.conversation_models import ConversationRoundtrip
from request_orchestrator.models.agent_result import AgentResult
from conversation.repository.repo_factory import get_conversation_repo
from conversation.summary_service import rebuild_conversation_summaries
from rendering.feedback import render_feedback_controls
from rendering.rendering import render_assistant_content, format_timestamp, _format_roundtrip_usage_summary
from common.config import (
    CONTENT_KEY,
    ROLE_ASSISTANT,
    ROLE_KEY,
    ROLE_USER,
    SUMMARY_BATCH_SIZE,
    SUMMARY_TRIGGER_SIZE,
)


MESSAGE_HISTORY_LIMIT = 10


def _build_answer_payload(answer: AgentResult) -> dict:
    payload = {
        "response": answer.raw_response,
        "result": [block.model_dump() for block in answer.answer_blocks],
        "used_evidence_ids": list(answer.used_evidence_ids),
        "hydrated_evidence_by_id": {
            evidence_id: evidence.model_dump()
            for evidence_id, evidence in answer.hydrated_evidence_by_id.items()
        },
        "next_question": answer.next_question,
        "roundtrip_summary": answer.roundtrip_summary,
        "tool_summary": answer.tool_summary,
        "agent_logs": answer.agent_logs,
    }
    roundtrip_latency_ms = getattr(answer, "roundtrip_latency_ms", None)
    if roundtrip_latency_ms is not None:
        payload["roundtrip_latency_ms"] = roundtrip_latency_ms
    return payload


def _merge_response_payload(roundtrip: ConversationRoundtrip, answer: AgentResult) -> dict:
    stored_payload = roundtrip.response_payload if isinstance(roundtrip.response_payload, dict) else {}
    answer_payload = _build_answer_payload(answer)
    merged = dict(stored_payload)

    for key, value in answer_payload.items():
        if key == "response":
            merged[key] = value or merged.get(key) or roundtrip.generated_response or ""
            continue
        if key == "agent_logs":
            merged[key] = value or merged.get(key) or {}
            continue
        if key == "result":
            merged[key] = value or merged.get(key) or []
            continue
        if key == "hydrated_evidence_by_id":
            merged[key] = value or merged.get(key) or {}
            continue
        if key == "used_evidence_ids":
            merged[key] = value or merged.get(key) or []
            continue
        if key == "tool_summary":
            merged[key] = value or merged.get(key) or {}
            continue
        merged[key] = value if value not in ("", None) else merged.get(key)

    return merged


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
    answer: AgentResult,
    roundtrip: ConversationRoundtrip,
) -> None:
    conversation_repository = get_conversation_repo()

    payload = _merge_response_payload(roundtrip, answer)
    rendered_response = str(payload.get("response") or answer.raw_response or roundtrip.generated_response or "")

    now = datetime.now(timezone.utc)
    assistant_message = {
        ROLE_KEY: ROLE_ASSISTANT,
        CONTENT_KEY: rendered_response,
        "payload": payload,
        "timestamp": now,
        "roundtrip_id": str(roundtrip.id),
        "model": roundtrip.model,
        "feedback_id": None,
    }
    st.session_state.messages.append(assistant_message)
    with st.chat_message(ROLE_ASSISTANT):
        render_assistant_content(rendered_response, payload)
        render_feedback_controls(
            roundtrip_id=roundtrip.id,
            model=roundtrip.model,
            sources_payload=payload,
            feedback_id=None,
            timestamp=format_timestamp(now),
            usage_summary=_format_roundtrip_usage_summary(payload.get("llm_usage") if isinstance(payload, dict) else None),
        )

    _update_conversation_summary(conversation_id, roundtrip)

    if roundtrip.message_index == 0:
        conversation_repository.set_conversation_title(
            conversation_id,
            generate_conversation_title(user_query),
        )
        st.rerun()
