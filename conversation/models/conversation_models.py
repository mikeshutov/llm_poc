from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator
from enum import StrEnum

from llm.conversation_model_config import ConversationModelConfig
from personalization.tone.models import TonePreferences


class ConversationSummaryResponse(BaseModel):
    conversation_summary: str = ""

    @field_validator("conversation_summary", mode="before")
    @classmethod
    def coerce_list_to_str(cls, v):
        if isinstance(v, list):
            return "\n".join(str(item) for item in v)
        return v


class ConversationContext(BaseModel):
    conversation_summary: str = ""
    latest_conversation_summary: str = ""
    # The ordered persistence message type is retained so callers can move
    # between repository, context, and replay views without adapters. Internal
    # tool messages are filtered by the prompt or rendering layer as needed.
    messages: list[ConversationMessage] = Field(default_factory=list)


class ConversationMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sources: list[str] = Field(default_factory=list)
    source_conversation_id: UUID | None = None
    source_roundtrip_id: UUID | None = None
    source_message_index: int | None = None


class ConversationMessageKind(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


class ConversationMessageVisibility(StrEnum):
    VISIBLE = "visible"
    INTERNAL = "internal"


@dataclass(frozen=True)
class NodeLog:
    id: UUID
    scope_type: str
    scope_id: UUID
    sequence: int
    node_name: str
    stage: str
    event_type: str
    level: str
    message: str
    payload: dict[str, Any]
    duration_ms: int | None
    correlation_id: UUID | None
    created_at: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", dict(self.payload) if isinstance(self.payload, dict) else {})


@dataclass(frozen=True)
class ConversationMessage:
    id: UUID
    roundtrip_id: UUID
    sequence: int
    kind: str
    visibility: str
    role: str
    content: str
    payload: dict[str, Any]
    metadata: dict[str, Any]
    parent_message_id: UUID | None
    created_at: str
    embedding: list[float] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", dict(self.payload) if isinstance(self.payload, dict) else {})
        object.__setattr__(self, "metadata", dict(self.metadata) if isinstance(self.metadata, dict) else {})


ConversationContext.model_rebuild()


@dataclass(frozen=False)
class Conversation:
    id: UUID
    user_id: str
    title: Optional[str]
    created_at: str
    metadata: ConversationMetadata
    tone_state: TonePreferences
    summary: str = ""
    summary_embedding: Optional[list[float]] = None

    def __post_init__(self) -> None:
        self.metadata = ConversationMetadata.model_validate(self.metadata)
        self.tone_state = TonePreferences.model_validate(self.tone_state)


@dataclass(frozen=False)
class ConversationRoundtrip:
    id: UUID
    conversation_id: UUID
    message_index: int
    roundtrip_summary: Optional[str] = None
    roundtrip_summary_embedding: Optional[list[float]] = None
    created_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"

    def __post_init__(self) -> None:
        self.metadata = dict(self.metadata) if isinstance(self.metadata, dict) else {}


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
class ConversationSummary:
    id: UUID
    conversation_id: UUID
    summary: str
    message_index_cutoff: int
    created_at: str


@dataclass(frozen=False)
class ConversationMemory:
    conversation_id: UUID
    summary: str
    last_used_date: str
    relevance_score: float
    matched_roundtrip_id: UUID | None = None
    matched_message_id: UUID | None = None


@dataclass(frozen=False)
class RoundtripMemory:
    conversation_id: UUID
    roundtrip_id: UUID
    message_index: int
    roundtrip_summary: str
    created_at: str
    relevance_score: float
    messages: list[ConversationMessage] = field(default_factory=list)
    matched_message_id: UUID | None = None


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
    user_id: Optional[str]
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
