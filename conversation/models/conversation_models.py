from dataclasses import dataclass
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class ConversationSummaryResponse(BaseModel):
    conversation_summary: str = ""
    tool_summary: str = ""

    @field_validator("conversation_summary", "tool_summary", mode="before")
    @classmethod
    def coerce_list_to_str(cls, v):
        if isinstance(v, list):
            return "\n".join(str(item) for item in v)
        return v


class ToolSummaryContext(BaseModel):
    used_tools: list[str] = Field(default_factory=list)
    produced: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    freshness: str = ""


class RecentRoundtripSummary(BaseModel):
    message_index: int
    roundtrip_summary: str


class RecentRoundtripToolSummary(BaseModel):
    message_index: int
    tool_summary: ToolSummaryContext


class ConversationContext(BaseModel):
    conversation_summary: str = ""
    tool_summary: str = ""
    recent_roundtrip_summaries: list[RecentRoundtripSummary] = Field(default_factory=list)
    recent_roundtrip_tool_summaries: list[RecentRoundtripToolSummary] = Field(default_factory=list)


@dataclass(frozen=False)
class Conversation:
    id: UUID
    user_id: str
    title: Optional[str]
    created_at: str
    metadata: dict[str, Any]
    tone_state: dict[str, Any]
    summary: str = ""


@dataclass(frozen=False)
class ConversationRoundtrip:
    id: UUID
    conversation_id: UUID
    message_index: int
    user_prompt: str
    generated_response: str
    roundtrip_summary: Optional[str]
    roundtrip_summary_embedding: Optional[list[float]]
    response_payload: dict[str, Any]
    parsed_query: dict[str, Any]
    created_at: str
    metadata: dict[str, Any]
    model: Optional[str] = None
    feedback_id: Optional[UUID] = None


@dataclass(frozen=False)
class RoundtripFeedback:
    id: UUID
    roundtrip_id: UUID
    met_expectation: bool
    reason: Optional[str]
    expected_answer: Optional[str]
    created_at: str
    model: Optional[str] = None


@dataclass(frozen=False)
class RoundtripPrompt:
    id: UUID
    roundtrip_id: UUID
    agent: str
    prompt_step: str
    prompt: str
    created_at: str


@dataclass(frozen=False)
class ConversationSummary:
    id: UUID
    conversation_id: UUID
    summary: str
    tool_summary: str
    message_index_cutoff: int
    created_at: str
