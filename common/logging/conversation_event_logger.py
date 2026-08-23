from __future__ import annotations

from typing import Any
from uuid import UUID

from conversation.repository.repo_factory import get_conversation_repo
from request_orchestrator.shared.runtime_context import (
    get_current_conversation_id,
    get_current_roundtrip_id,
)


def create_conversation_event(
    *,
    event_type: str,
    source: str,
    conversation_id: str | None = None,
    roundtrip_id: UUID | str | None = None,
    agent_name: str = "",
    node_name: str = "",
    iteration: int | None = None,
    payload: dict[str, Any] | None = None,
) -> None:
    resolved_conversation_id = conversation_id or get_current_conversation_id()
    if not resolved_conversation_id:
        return

    resolved_roundtrip_id = roundtrip_id
    if resolved_roundtrip_id is None:
        resolved_roundtrip_id = get_current_roundtrip_id()

    try:
        get_conversation_repo().create_conversation_event(
            conversation_id=UUID(resolved_conversation_id),
            roundtrip_id=(
                None
                if not resolved_roundtrip_id
                else UUID(str(resolved_roundtrip_id))
            ),
            event_type=event_type,
            source=source,
            agent_name=agent_name,
            node_name=node_name,
            iteration=iteration,
            payload={} if payload is None else dict(payload),
        )
    except Exception:
        return


def log_roundtrip_prompt(
    *,
    roundtrip_id: UUID | None,
    agent: str,
    prompt_step: str,
    prompt: str,
) -> None:
    if roundtrip_id is not None:
        try:
            get_conversation_repo().create_roundtrip_prompt(
                roundtrip_id,
                agent=agent,
                prompt_step=prompt_step,
                prompt=prompt,
            )
        except Exception:
            pass

    create_conversation_event(
        event_type="prompt",
        source=agent,
        agent_name=agent,
        node_name=prompt_step,
        payload={
            "prompt_step": prompt_step,
            "prompt": prompt,
        },
    )
