from uuid import UUID

from llm.llm_message_builder import compose_messages_from_roundtrips
from common.message_constants import CONTENT_KEY, ROLE_KEY, ROLE_SYSTEM, ROLE_USER


def build_roundtrip_context(conversation_repository, conversation_id: str, user_query: str, limit: int = 5):
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
