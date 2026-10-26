from __future__ import annotations

from uuid import UUID

from common.config import SUMMARY_BATCH_SIZE, SUMMARY_TRIGGER_SIZE
from conversation.constants import SOURCE_REPLAY
from conversation.models.conversation_models import ConversationMetadata
from conversation.models.replay_models import PopulatedReplayConversation, PreparedReplayConversation
from conversation.repository.repo_factory import get_conversation_repo
from conversation.summary_service import rebuild_conversation_summaries


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
        metadata=ConversationMetadata(
            sources=[SOURCE_REPLAY],
            source_conversation_id=source_conversation.id,
            source_roundtrip_id=source_roundtrip.id,
            source_message_index=source_roundtrip.message_index,
        ),
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
        user_prompt=next(
            (message.content for message in repo.list_messages(source_roundtrip.id, include_internal=False)
             if message.kind == "user"),
            "",
        ),
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

    # A replay conversation is a materialized snapshot of the source prefix.
    # Reconcile that prefix before copying so destinations created by the old
    # replay implementation cannot contribute stale or duplicated turns.
    expected_source_indices = {roundtrip.message_index for roundtrip in source_roundtrips}
    expected_destination_indices = {
        *expected_source_indices,
        prepared_replay.source_message_index,
    }
    existing_roundtrips = repo.list_roundtrips(
        parsed_conversation_id,
        limit=1000,
        newest_first=False,
    )
    retained_indices: set[int] = set()
    for existing_roundtrip in existing_roundtrips:
        if (
            existing_roundtrip.message_index not in expected_destination_indices
            or existing_roundtrip.message_index in retained_indices
        ):
            repo.delete_roundtrip(existing_roundtrip.id)
            continue
        retained_indices.add(existing_roundtrip.message_index)

    # Streamlit reruns and retries can enter this function more than once for
    # the same prepared replay. The destination is replay-only, so a matching
    # message index means that historical roundtrip has already been copied.
    existing_roundtrips = repo.list_roundtrips(
        parsed_conversation_id,
        limit=1000,
        newest_first=False,
    )
    existing_message_indices = {roundtrip.message_index for roundtrip in existing_roundtrips}

    roundtrip_id_map: dict[UUID, UUID] = {}
    copied_source_indices: set[int] = set()
    for historical_roundtrip in source_roundtrips:
        if (
            historical_roundtrip.message_index in existing_message_indices
            or historical_roundtrip.message_index in copied_source_indices
        ):
            continue
        copied_roundtrip = repo.copy_roundtrip_to_conversation(parsed_conversation_id, historical_roundtrip)
        roundtrip_id_map[historical_roundtrip.id] = copied_roundtrip.id
        copied_source_indices.add(historical_roundtrip.message_index)

    rebuild_conversation_summaries(
        str(parsed_conversation_id),
        summary_batch_size=SUMMARY_BATCH_SIZE,
        summary_trigger_size=SUMMARY_TRIGGER_SIZE,
    )

    return PopulatedReplayConversation(
        conversation_id=str(parsed_conversation_id),
        user_prompt=prepared_replay.user_prompt,
    )


def replay_target_completed(
    prepared_replay: PreparedReplayConversation,
) -> bool:
    """Return whether the selected source turn already ran in the replay.

    Replay execution is deliberately idempotent.  Checking the replay turn's
    persisted boundary, rather than comparing rendered prompt text, prevents
    Streamlit retries from executing the same target turn twice while still
    preserving legitimate repeated user prompts.
    """
    repo = get_conversation_repo()
    conversation_id = UUID(prepared_replay.conversation_id)
    for roundtrip in repo.list_roundtrips(
        conversation_id,
        limit=1000,
        newest_first=False,
    ):
        if (
            roundtrip.message_index != prepared_replay.source_message_index
            or roundtrip.status != "completed"
        ):
            continue
        if any(
            message.kind == "user"
            for message in repo.list_messages(roundtrip.id, include_internal=False)
        ):
            return True
    return False
