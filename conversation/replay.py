from __future__ import annotations

from uuid import UUID

from common.config import SUMMARY_BATCH_SIZE, SUMMARY_TRIGGER_SIZE
from conversation.models.replay_models import PopulatedReplayConversation, PreparedReplayConversation
from conversation.repository.repo_factory import get_conversation_repo
from conversation.summary_service import rebuild_conversation_summaries
from tool.repository.tool_call_repository import ToolCallRepository


def _resolve_replay_source(
    roundtrip_id: str | UUID,
    *,
    user_id: str | None = None,
):
    repo = get_conversation_repo()
    parsed_roundtrip_id = UUID(str(roundtrip_id))
    source_roundtrip = repo.get_roundtrip(parsed_roundtrip_id)
    if source_roundtrip is None:
        raise ValueError(f"Roundtrip {parsed_roundtrip_id} was not found.")

    source_conversation = repo.get_conversation(source_roundtrip.conversation_id)
    if source_conversation is None:
        raise ValueError(f"Conversation {source_roundtrip.conversation_id} was not found.")
    if user_id is not None and user_id != source_conversation.user_id:
        raise ValueError(
            f"Roundtrip {parsed_roundtrip_id} belongs to user {source_conversation.user_id}, not {user_id}"
        )
    return repo, source_roundtrip, source_conversation


def prepare_replay(roundtrip_id: str | UUID, user_id: str) -> PreparedReplayConversation:
    repo, source_roundtrip, source_conversation = _resolve_replay_source(roundtrip_id, user_id=user_id)

    new_conversation = repo.create_conversation(
        user_id=user_id,
        metadata={
            "source": "replay",
            "source_conversation_id": str(source_conversation.id),
            "source_roundtrip_id": str(source_roundtrip.id),
            "source_message_index": source_roundtrip.message_index,
        },
    )

    if source_conversation.tone_state:
        repo.update_tone_state(new_conversation.id, source_conversation.tone_state)

    source_title = (source_conversation.title or "").strip()
    if source_title:
        replay_title = source_title if source_title.endswith("(Replay)") else f"{source_title} (Replay)"
        repo.set_conversation_title(str(new_conversation.id), replay_title)

    return PreparedReplayConversation(
        conversation_id=str(new_conversation.id),
        source_roundtrip_id=str(source_roundtrip.id),
        source_conversation_id=str(source_roundtrip.conversation_id),
        source_message_index=source_roundtrip.message_index,
        user_prompt=source_roundtrip.user_prompt,
    )


def execute_replay(
    prepared_replay: PreparedReplayConversation,
) -> PopulatedReplayConversation:
    repo = get_conversation_repo()
    parsed_conversation_id = UUID(prepared_replay.conversation_id)
    source_conversation_id = UUID(prepared_replay.source_conversation_id)

    history_cutoff = prepared_replay.source_message_index - 1
    source_roundtrips = (
        repo.list_roundtrips_through_message_index(
            source_conversation_id,
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

    return PopulatedReplayConversation(
        conversation_id=str(parsed_conversation_id),
        user_prompt=prepared_replay.user_prompt,
    )
