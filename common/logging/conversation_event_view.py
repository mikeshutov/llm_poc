from __future__ import annotations

from typing import Any
from uuid import UUID

from conversation.models.conversation_models import ConversationEvent
from conversation.repository.repo_factory import get_conversation_repo

DISPLAY_EXCLUDED_EVENT_TYPES = {"prompt", "llm_call"}


def normalize_conversation_event(event: ConversationEvent) -> tuple[str, dict[str, Any]]:
    payload = dict(event.payload or {})
    if event.event_type == "llm_call":
        model_scope = payload.get("model_scope")
        if not isinstance(model_scope, str) or not model_scope.strip():
            payload["model_scope"] = payload.get("agent") or event.agent_name
        owner_agent_name = payload.get("owner_agent_name")
        if isinstance(owner_agent_name, str) and owner_agent_name.strip():
            agent_name = owner_agent_name.strip()
        else:
            agent_name = event.agent_name.strip() or event.source.strip() or "event"
    else:
        agent_name = event.agent_name.strip() or event.source.strip() or "event"

    payload.setdefault("agent_name", agent_name)
    payload.setdefault("kind", event.event_type)
    if event.node_name.strip():
        payload.setdefault("node_name", event.node_name)
    if event.iteration is not None:
        payload.setdefault("iteration", event.iteration)
    return agent_name, payload


def fetch_agent_logs_for_roundtrip(roundtrip_id: str | None) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(roundtrip_id, str) or not roundtrip_id.strip():
        return {}
    try:
        events = get_conversation_repo().list_conversation_events_for_roundtrip(UUID(roundtrip_id))
    except Exception:
        return {}

    grouped: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        if event.event_type in DISPLAY_EXCLUDED_EVENT_TYPES:
            continue
        agent_name, payload = normalize_conversation_event(event)
        grouped.setdefault(agent_name, []).append(payload)
    return grouped


def fetch_llm_call_payloads_for_roundtrip(roundtrip_id: str | None) -> list[dict[str, Any]]:
    if not isinstance(roundtrip_id, str) or not roundtrip_id.strip():
        return []
    parsed_roundtrip_id = UUID(roundtrip_id)
    repo = get_conversation_repo()
    try:
        events = repo.list_conversation_events_for_roundtrip(parsed_roundtrip_id)
    except Exception:
        return []

    llm_calls: list[dict[str, Any]] = []
    for event in events:
        if event.event_type != "llm_call":
            continue
        if not isinstance(event.payload, dict):
            continue
        llm_calls.append(dict(event.payload))
    if llm_calls and all(_llm_call_payload_has_trace_data(payload) for payload in llm_calls):
        return llm_calls
    try:
        records = repo.list_llm_calls_for_roundtrip(parsed_roundtrip_id)
    except Exception:
        return llm_calls
    if records:
        return list(records)
    return llm_calls


def _llm_call_payload_has_trace_data(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    if payload.get("input_object") is not None or payload.get("output_object") is not None:
        return True
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return False
    return metadata.get("input_object") is not None or metadata.get("output_object") is not None
