import json
from uuid import UUID

from conversation.models.conversation_models import (
    ConversationMessage,
    ConversationContext,
)
from conversation.repository.repo_factory import get_conversation_repo
from common.data import bounded_json_value

MAX_CONTEXT_MESSAGES = 120
MAX_CONTEXT_BYTES = 120_000


def _source_evidence_ids_from_message(message: ConversationMessage) -> set[str]:
    payload = message.payload if isinstance(message.payload, dict) else {}
    input_payload = payload.get("input_payload")
    if not isinstance(input_payload, dict):
        return set()
    source_evidence_ids = input_payload.get("source_evidence_ids", [])
    if not isinstance(source_evidence_ids, list):
        return set()
    return {
        evidence_id.strip()
        for evidence_id in source_evidence_ids
        if isinstance(evidence_id, str) and evidence_id.strip()
    }


def _project_context_message(
    message: ConversationMessage,
    *,
    suppressed_evidence_ids: set[str] | None = None,
) -> dict:
    kind = str(message.kind)
    visibility = str(message.visibility)
    if message.is_subagent:
        return {}
    if visibility != "visible" and kind not in {"tool_call", "tool_result"}:
        return {}
    if kind in {"tool_call", "tool_result"}:
        payload = dict(message.payload or {})
        payload.pop("output_payload", None)
        if isinstance(payload.get("evidence"), list):
            compactable = payload.get("compactable", True) is not False
            payload["evidence"] = [
                bounded_json_value(
                    item,
                    max_bytes=8_000,
                ) if isinstance(item, dict) else item
                for item in payload["evidence"]
                if not compactable or not (
                    suppressed_evidence_ids
                    and isinstance(item, dict)
                    and str(item.get("id", "")).strip() in suppressed_evidence_ids
                )
            ]
    else:
        payload = {
            key: message.payload.get(key)
            for key in ("next_question", "used_evidence_ids")
            if key in message.payload
        }
    return {
        "id": message.id,
        "roundtrip_id": message.roundtrip_id,
        "sequence": message.sequence,
        "kind": message.kind,
        "visibility": message.visibility,
        "role": message.role,
        "content": message.content,
        "payload": bounded_json_value(payload, max_bytes=16_000),
        "metadata": bounded_json_value(message.metadata, max_bytes=2_000),
        "parent_message_id": message.parent_message_id,
        "created_at": str(message.created_at),
        "embedding": None,
    }


def build_roundtrip_context(conversation_id: str, limit: int = 5) -> ConversationContext:
    # Streamlit can reload this module independently from the model module.
    # Rebuild the Pydantic schema so ConversationContext accepts the current
    # ConversationMessage class rather than a class object from a prior reload.
    ConversationContext.model_rebuild(force=True)
    conversation_repository = get_conversation_repo()
    resolved_conversation_id = UUID(conversation_id)
    conversation = conversation_repository.get_conversation(resolved_conversation_id)
    latest_summary = conversation_repository.get_latest_summary(resolved_conversation_id)
    after_index = latest_summary.message_index_cutoff if latest_summary else None
    conversation_roundtrips = conversation_repository.list_roundtrips(
        resolved_conversation_id,
        limit=limit,
        after_message_index=after_index,
        newest_first=True,
    )
    # Keep the complete ordered message stream for model callers.  The
    # visible subset is derived separately for summaries and chat history.
    messages: list[dict] = []
    # list_roundtrips returns only completed roundtrips. ConversationRepository
    # returns this result in chronological order even
    # when newest_first=True is used to select the most recent window. Keep
    # the selected window in oldest-to-newest order for the LLM message list.
    selected_messages: list[ConversationMessage] = []
    for roundtrip in conversation_roundtrips:
        roundtrip_messages = conversation_repository.list_messages(
            roundtrip.id,
            include_internal=True,
            exclude_subagents=True,
        )
        for message in roundtrip_messages:
            if not message.content.strip() and not message.payload:
                continue
            selected_messages.append(message)

    # A workspace write/edit records which evidence it has condensed into the
    # file. Collect those links before projecting messages so source evidence is
    # suppressed even when its tool result appears before the write/edit.
    suppressed_evidence_ids = {
        evidence_id
        for message in selected_messages
        for evidence_id in _source_evidence_ids_from_message(message)
    }
    for message in selected_messages:
        projected = _project_context_message(
            message,
            suppressed_evidence_ids=suppressed_evidence_ids,
        )
        if projected:
            messages.append(projected)

    while len(messages) > MAX_CONTEXT_MESSAGES:
        messages.pop(0)
    while messages:
        rendered_size = len(json.dumps(messages, default=str).encode("utf-8"))
        if rendered_size <= MAX_CONTEXT_BYTES:
            break
        messages.pop(0)

    return ConversationContext(
        conversation_summary=conversation.summary if conversation else "",
        latest_conversation_summary=latest_summary.summary if latest_summary else "",
        messages=messages,
    )
