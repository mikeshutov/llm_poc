from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, field_validator


class ConversationSummaryResponse(BaseModel):
    conversation_summary: str = ""
    tool_summary: str = ""

    @field_validator("conversation_summary", "tool_summary", mode="before")
    @classmethod
    def coerce_list_to_str(cls, v):
        if isinstance(v, list):
            return "\n".join(str(item) for item in v)
        return v


@dataclass(frozen=False)
class Conversation:
    id: UUID
    user_id: str
    title: Optional[str]
    created_at: str  
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
    created_at: str  
    metadata: dict[str, Any]

@dataclass(frozen=False)
class ConversationSummary:
    id: UUID
    conversation_id: UUID
    summary: str
    tool_summary: str
    message_index_cutoff: int
    created_at: str  
