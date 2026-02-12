from dataclasses import dataclass
from typing import Any, Optional
from uuid import UUID


@dataclass(frozen=False)
class Conversation:
    id: UUID
    user_id: str
    title: Optional[str]
    created_at: str  # or datetime
    metadata: dict[str, Any]
    tone_state: dict[str, Any]


@dataclass(frozen=False)
class ConversationRoundtrip:
    id: UUID
    conversation_id: UUID
    message_index: int
    user_prompt: str
    generated_response: str
    response_payload: dict[str, Any]
    parsed_query: dict[str, Any]
    created_at: str  # or datetime
    metadata: dict[str, Any]


@dataclass(frozen=False)
class ConversationSummary:
    id: UUID
    conversation_id: UUID
    summary: str
    message_index_cutoff: int
    created_at: str  # or datetime
