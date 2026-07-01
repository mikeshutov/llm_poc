from uuid import UUID

from conversation.models.conversation_models import (
    ConversationContext,
    RecentRoundtripSummary,
    RecentRoundtripToolSummary,
    ToolSummaryContext,
)
from conversation.repository.repo_factory import get_conversation_repo


def build_roundtrip_context(conversation_id: str, limit: int = 5) -> ConversationContext:
    conversation_repository = get_conversation_repo()
    conversation = conversation_repository.get_conversation(UUID(conversation_id))
    latest_summary = conversation_repository.get_latest_summary(UUID(conversation_id))
    after_index = latest_summary.message_index_cutoff if latest_summary else None
    conversation_roundtrips = conversation_repository.list_roundtrips(
        UUID(conversation_id),
        limit=limit,
        after_message_index=after_index,
    )

    recent_roundtrip_summaries = [
        RecentRoundtripSummary(
            message_index=rt.message_index,
            roundtrip_summary=rt.roundtrip_summary,
        )
        for rt in conversation_roundtrips
        if rt.roundtrip_summary
    ]

    recent_roundtrip_tool_summaries = []
    for rt in conversation_roundtrips:
        payload = rt.response_payload if isinstance(rt.response_payload, dict) else {}
        tool_summary = payload.get("tool_summary")
        if not tool_summary:
            continue
        recent_roundtrip_tool_summaries.append(
            RecentRoundtripToolSummary(
                message_index=rt.message_index,
                tool_summary=ToolSummaryContext.model_validate(tool_summary),
            )
        )

    return ConversationContext(
        conversation_summary=conversation.summary if conversation else "",
        tool_summary=latest_summary.tool_summary if latest_summary else "",
        recent_roundtrip_summaries=recent_roundtrip_summaries,
        recent_roundtrip_tool_summaries=recent_roundtrip_tool_summaries,
    )
