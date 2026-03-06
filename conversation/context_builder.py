from uuid import UUID

from conversation.repository.repo_factory import get_conversation_repo
from rendering.messages.compose import compose_messages_from_roundtrips
from common.message_constants import CONTENT_KEY, ROLE_KEY, ROLE_SYSTEM, ROLE_USER


def build_roundtrip_context(conversation_id: str, user_query: str, limit: int = 5):
    conversation_repository = get_conversation_repo()
    latest_summary = conversation_repository.get_latest_summary(UUID(conversation_id))
    after_index = latest_summary.message_index_cutoff if latest_summary else None
    conversation_roundtrips = conversation_repository.list_roundtrips(
        UUID(conversation_id),
        limit=limit,
        after_message_index=after_index,
    )
    roundtrips_with_latest = [
        *(
            [{ROLE_KEY: ROLE_SYSTEM, CONTENT_KEY: f"Conversation summary:\n{latest_summary.summary}"}]
            if latest_summary
            else []
        ),
        *compose_messages_from_roundtrips(conversation_roundtrips),
        {ROLE_KEY: ROLE_USER, CONTENT_KEY: user_query},
    ]
    return roundtrips_with_latest
