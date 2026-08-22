from __future__ import annotations

from uuid import UUID

from conversation.conversation import generate_conversation_summary
from conversation.models.conversation_models import ConversationRoundtrip
from conversation.repository.repo_factory import get_conversation_repo
from llm.clients.embeddings import embed_text
from tool.repository.tool_call_repository import ToolCallRepository


def _update_top_level_conversation_summary(
    conversation_id: str,
    *,
    previous_summary: str,
    unsummarized_roundtrips: list[ConversationRoundtrip],
    tool_calls_by_roundtrip,
) -> None:
    if not previous_summary and not unsummarized_roundtrips:
        return

    conversation_repository = get_conversation_repo()
    if unsummarized_roundtrips:
        top_level_summary = generate_conversation_summary(
            unsummarized_roundtrips,
            tool_call_map=tool_calls_by_roundtrip,
            previous_summary=previous_summary or None,
        )
        summary_text = top_level_summary.conversation_summary.strip()
    else:
        summary_text = previous_summary.strip()
    summary_embedding = embed_text(summary_text) if summary_text else None
    conversation_repository.update_conversation_summary(
        UUID(conversation_id),
        summary_text,
        summary_embedding=summary_embedding,
    )


def _update_batched_conversation_summaries(conversation_id: str, summary_batch_size: int, summary_trigger_size: int) -> None:
    conversation_repository = get_conversation_repo()
    latest_summary = conversation_repository.get_latest_summary(UUID(conversation_id))
    last_cutoff = latest_summary.message_index_cutoff if latest_summary else -1
    previous_batch_summary = latest_summary.summary if latest_summary else ""

    while True:
        unsummarized_roundtrips = conversation_repository.list_roundtrips(
            UUID(conversation_id),
            limit=summary_trigger_size,
            after_message_index=last_cutoff,
        )
        if len(unsummarized_roundtrips) < summary_trigger_size:
            return

        batch_roundtrips = unsummarized_roundtrips[:summary_batch_size]
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


def rebuild_conversation_summaries(
    conversation_id: str,
    *,
    summary_batch_size: int,
    summary_trigger_size: int,
) -> None:
    conversation_repository = get_conversation_repo()
    _update_batched_conversation_summaries(
        conversation_id,
        summary_batch_size=summary_batch_size,
        summary_trigger_size=summary_trigger_size,
    )
    latest_summary = conversation_repository.get_latest_summary(UUID(conversation_id))
    last_cutoff = latest_summary.message_index_cutoff if latest_summary else -1
    unsummarized_roundtrips = conversation_repository.list_roundtrips(
        UUID(conversation_id),
        limit=summary_trigger_size,
        after_message_index=last_cutoff,
    )
    roundtrip_ids = [rt.id for rt in unsummarized_roundtrips]
    tool_calls_by_roundtrip = ToolCallRepository().get_tool_calls_by_roundtrips(roundtrip_ids)
    _update_top_level_conversation_summary(
        conversation_id,
        previous_summary="" if latest_summary is None else latest_summary.summary,
        unsummarized_roundtrips=unsummarized_roundtrips,
        tool_calls_by_roundtrip=tool_calls_by_roundtrip,
    )
