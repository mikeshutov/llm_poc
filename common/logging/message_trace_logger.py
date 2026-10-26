from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from conversation.repository.repo_factory import get_conversation_repo
from common.data import bounded_json_value
from request_orchestrator.constants import NodeName
from request_orchestrator.shared.runtime_context import get_current_roundtrip_id

logger = logging.getLogger(__name__)


def _node_value(node_name: NodeName | str) -> str:
    return node_name.value if isinstance(node_name, NodeName) else str(node_name)


def compact_trace_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    return {
        "argument_types": {key: type(value).__name__ for key, value in inputs.items()},
        "task": next(
            (str(getattr(value, "task", ""))[:500] for value in inputs.values() if hasattr(value, "task")),
            "",
        ),
    }


def compact_trace_outputs(outputs: Any) -> dict[str, Any]:
    return {"output_type": type(outputs).__name__}


def record_node_log(
    *,
    node_name: NodeName | str,
    event_type: str,
    stage: str = "",
    level: str = "info",
    message: str = "",
    payload: dict[str, Any] | None = None,
    duration_ms: int | None = None,
    roundtrip_id: UUID | str | None = None,
    conversation_id: str | None = None,
    source: str = "",
    agent_name: str = "",
    agent_version: int | None = None,
    iteration: int | None = None,
) -> None:
    """Persist a bounded node diagnostic without entering conversation history."""
    resolved_node_name = _node_value(node_name)
    resolved_roundtrip_id = roundtrip_id or get_current_roundtrip_id()
    if not resolved_roundtrip_id:
        return
    try:
        resolved_payload = bounded_json_value({} if payload is None else dict(payload))
        trace_data = (
            resolved_payload.get("data", resolved_payload)
            if isinstance(resolved_payload, dict)
            else resolved_payload
        )
        repository = get_conversation_repo()
        repository.append_node_log(
            "roundtrip",
            UUID(str(resolved_roundtrip_id)),
            node_name=resolved_node_name,
            stage=stage or resolved_node_name,
            event_type=event_type,
            level=level,
            message=message,
            duration_ms=duration_ms,
            payload=bounded_json_value({
                "source": source or resolved_node_name,
                "agent_name": agent_name or resolved_node_name,
                "node_name": resolved_node_name,
                "iteration": iteration,
                "agent_version": agent_version,
                "data": trace_data,
            }),
        )
    except Exception:
        logger.exception(
            "Unable to persist node log: node=%s event=%s roundtrip=%s",
            resolved_node_name,
            event_type,
            resolved_roundtrip_id,
        )


def record_node_success(
    *,
    node_name: NodeName | str,
    event_type: str,
    message: str,
    payload: dict[str, Any] | None = None,
    duration_ms: int | None = None,
    stage: str = "",
    roundtrip_id: UUID | str | None = None,
) -> None:
    record_node_log(
        node_name=node_name,
        event_type=event_type,
        stage=stage,
        message=message,
        payload=payload,
        duration_ms=duration_ms,
        roundtrip_id=roundtrip_id,
    )


def record_node_failure(
    *,
    node_name: NodeName | str,
    event_type: str,
    message: str,
    error: Exception | str,
    payload: dict[str, Any] | None = None,
    duration_ms: int | None = None,
    stage: str = "",
    roundtrip_id: UUID | str | None = None,
) -> None:
    record_node_log(
        node_name=node_name,
        event_type=event_type,
        stage=stage,
        level="error",
        message=message,
        payload={**(payload or {}), "error": str(error)},
        duration_ms=duration_ms,
        roundtrip_id=roundtrip_id,
    )
