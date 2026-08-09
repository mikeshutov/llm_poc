from dataclasses import dataclass
from decimal import Decimal
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


class RecentRoundtrip(BaseModel):
    message_index: int
    user_prompt: str = ""
    roundtrip_summary: str = ""


class RecentRoundtripToolSummary(BaseModel):
    message_index: int
    tool_summary: ToolSummaryContext


class ConversationContext(BaseModel):
    conversation_summary: str = ""
    latest_conversation_summary: str = ""
    tool_summary: str = ""
    recent_roundtrips: list[RecentRoundtrip] = Field(default_factory=list)
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
    summary_embedding: Optional[list[float]] = None


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


@dataclass(frozen=False)
class ConversationMemory:
    conversation_id: UUID
    summary: str
    last_used_date: str
    relevance_score: float


@dataclass(frozen=False)
class RoundtripMemory:
    conversation_id: UUID
    roundtrip_id: UUID
    message_index: int
    user_prompt: str
    generated_response: str
    roundtrip_summary: str
    created_at: str
    relevance_score: float


@dataclass(frozen=False)
class LlmUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_input_tokens: int = 0


@dataclass(frozen=False)
class LlmCallRecord:
    id: UUID
    conversation_id: Optional[UUID]
    roundtrip_id: Optional[UUID]
    agent: Optional[str]
    stage: Optional[str]
    callsite: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_input_tokens: int
    input_price_per_million_tokens: Decimal
    output_price_per_million_tokens: Decimal
    computed_input_cost: Decimal
    computed_output_cost: Decimal
    computed_total_cost: Decimal
    metadata: dict[str, Any]
    created_at: str
    updated_at: str
