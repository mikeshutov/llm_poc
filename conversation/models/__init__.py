from __future__ import annotations

from pydantic import BaseModel


class PreparedReplayConversation(BaseModel):
    conversation_id: str
    source_roundtrip_id: str
    source_conversation_id: str
    source_message_index: int
    user_prompt: str


class PopulatedReplayConversation(BaseModel):
    conversation_id: str
    user_prompt: str
