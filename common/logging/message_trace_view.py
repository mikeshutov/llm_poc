from __future__ import annotations

from typing import Any
from uuid import UUID

from conversation.repository.repo_factory import get_conversation_repo

def fetch_trace_messages_for_roundtrip(
    roundtrip_id: str | None,
    *,
    node_name: str | None = None,
    stage: str | None = None,
    event_type: str | None = None,
    level: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(roundtrip_id, str) or not roundtrip_id.strip():
        return {}
    try:
        repository = get_conversation_repo()
        logs = repository.list_node_logs(
            "roundtrip",
            UUID(roundtrip_id),
            node_name=node_name,
            stage=stage,
            event_type=event_type,
            level=level,
        )
    except Exception:
        return {}

    grouped: dict[str, list[dict[str, Any]]] = {}
    for log in logs:
        event_type = log.event_type
        payload = log.payload.get("data", log.payload)
        if not isinstance(payload, dict):
            payload = {"content": log.message}
        agent_name = log.node_name or "event"
        payload = dict(payload)
        payload.setdefault("agent_name", agent_name)
        payload.setdefault("kind", event_type)
        payload.setdefault("node_name", log.node_name)
        payload.setdefault("stage", log.stage)
        payload.setdefault("level", log.level)
        payload.setdefault("sequence", log.sequence)
        grouped.setdefault(agent_name, []).append(payload)
    return grouped


def fetch_llm_trace_for_roundtrip(roundtrip_id: str | None) -> list[dict[str, Any]]:
    if not isinstance(roundtrip_id, str) or not roundtrip_id.strip():
        return []
    parsed_roundtrip_id = UUID(roundtrip_id)
    repo = get_conversation_repo()
    try:
        logs = repo.list_node_logs("roundtrip", parsed_roundtrip_id, event_type="llm_call")
    except Exception:
        return []

    llm_calls: list[dict[str, Any]] = []
    for log in logs:
        payload = log.payload.get("data", log.payload)
        if not isinstance(payload, dict):
            continue
        llm_calls.append(dict(payload))
    return llm_calls
