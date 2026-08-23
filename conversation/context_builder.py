from uuid import UUID

from conversation.models.conversation_models import (
    ConversationContext,
    RecentRoundtrip,
    RecentRoundtripToolSummary,
    ToolSummaryContext,
)
from conversation.repository.repo_factory import get_conversation_repo


def build_roundtrip_context(conversation_id: str, limit: int = 5) -> ConversationContext:
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
    latest_completed_roundtrip = conversation_repository.get_latest_completed_roundtrip(
        resolved_conversation_id,
    )

    recent_roundtrips = [
        RecentRoundtrip(
            message_index=rt.message_index,
            user_prompt=rt.user_prompt or "",
            roundtrip_summary=rt.roundtrip_summary or "",
            assistant_follow_up=rt.assistant_follow_up or "",
        )
        for rt in conversation_roundtrips
        if rt.user_prompt or rt.roundtrip_summary
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
        latest_conversation_summary=latest_summary.summary if latest_summary else "",
        tool_summary=latest_summary.tool_summary if latest_summary else "",
        recent_roundtrips=recent_roundtrips,
        recent_roundtrip_tool_summaries=recent_roundtrip_tool_summaries,
        previous_user_request=(
            latest_completed_roundtrip.user_prompt
            if latest_completed_roundtrip is not None
            else ""
        ),
        latest_assistant_follow_up=(
            latest_completed_roundtrip.assistant_follow_up
            if latest_completed_roundtrip is not None
            else ""
        ),
    )
