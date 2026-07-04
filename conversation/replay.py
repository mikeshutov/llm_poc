from __future__ import annotations

from typing import Any
from uuid import UUID

from common.message_constants import SUMMARY_BATCH_SIZE, SUMMARY_TRIGGER_SIZE
from conversation.repository.repo_factory import get_conversation_repo
from conversation.summary_service import rebuild_conversation_summaries
from tool.repository.tool_call_repository import ToolCallRepository


def prepare_replay_conversation(roundtrip_id: str | UUID) -> dict[str, Any]:
    repo = get_conversation_repo()
    parsed_roundtrip_id = UUID(str(roundtrip_id))
    source_roundtrip = repo.get_roundtrip(parsed_roundtrip_id)
    if source_roundtrip is None:
        raise ValueError(f"Roundtrip {parsed_roundtrip_id} was not found.")

    source_conversation = repo.get_conversation(source_roundtrip.conversation_id)
    if source_conversation is None:
        raise ValueError(f"Conversation {source_roundtrip.conversation_id} was not found.")

    replay_metadata = {
        "source": "replay",
        "source_conversation_id": str(source_conversation.id),
        "source_roundtrip_id": str(source_roundtrip.id),
        "source_message_index": source_roundtrip.message_index,
    }
    new_conversation = repo.create_conversation(
        user_id=source_conversation.user_id,
        metadata=replay_metadata,
    )

    if source_conversation.tone_state:
        repo.update_tone_state(new_conversation.id, source_conversation.tone_state)

    source_title = (source_conversation.title or "").strip()
    if source_title:
        replay_title = source_title if source_title.endswith("(Replay)") else f"{source_title} (Replay)"
        repo.set_conversation_title(str(new_conversation.id), replay_title)

    return {
        "conversation_id": str(new_conversation.id),
        "source_roundtrip_id": str(source_roundtrip.id),
    }


def populate_replay_conversation(conversation_id: str | UUID, source_roundtrip_id: str | UUID) -> dict[str, Any]:
    repo = get_conversation_repo()
    parsed_roundtrip_id = UUID(str(source_roundtrip_id))
    parsed_conversation_id = UUID(str(conversation_id))

    source_roundtrip = repo.get_roundtrip(parsed_roundtrip_id)
    if source_roundtrip is None:
        raise ValueError(f"Roundtrip {parsed_roundtrip_id} was not found.")

    history_cutoff = source_roundtrip.message_index - 1
    source_roundtrips = (
        repo.list_roundtrips_through_message_index(
            source_roundtrip.conversation_id,
            history_cutoff,
        )
        if history_cutoff >= 0
        else []
    )

    roundtrip_id_map: dict[UUID, UUID] = {}
    for historical_roundtrip in source_roundtrips:
        copied_roundtrip = repo.copy_roundtrip_to_conversation(parsed_conversation_id, historical_roundtrip)
        roundtrip_id_map[historical_roundtrip.id] = copied_roundtrip.id

    if roundtrip_id_map:
        ToolCallRepository().copy_tool_calls(roundtrip_id_map)

    rebuild_conversation_summaries(
        str(parsed_conversation_id),
        summary_batch_size=SUMMARY_BATCH_SIZE,
        summary_trigger_size=SUMMARY_TRIGGER_SIZE,
    )

    return {
        "conversation_id": str(parsed_conversation_id),
        "user_prompt": source_roundtrip.user_prompt,
    }
