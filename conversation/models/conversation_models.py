from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from llm.conversation_model_config import ConversationModelConfig
from personalization.tone.models import TonePreferences
from request_orchestrator.models.orchestrator_payload import EvidenceProducedByTool, OrchestratorPayload
from request_orchestrator.models.relevant_evidence import RelevantEvidenceByTool


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
    evidence_produced: EvidenceProducedByTool = Field(default_factory=EvidenceProducedByTool.empty)
    freshness: str = ""


class RecentRoundtrip(BaseModel):
    message_index: int
    user_prompt: str = ""
    roundtrip_summary: str = ""
    assistant_follow_up: str = ""
    used_evidence_ids: list[str] = Field(default_factory=list)
    relevant_evidence: RelevantEvidenceByTool = Field(default_factory=RelevantEvidenceByTool.empty)


class RecentRoundtripToolSummary(BaseModel):
    message_index: int
    tool_summary: ToolSummaryContext


class ConversationContext(BaseModel):
    conversation_summary: str = ""
    latest_conversation_summary: str = ""
    tool_summary: str = ""
    recent_roundtrips: list[RecentRoundtrip] = Field(default_factory=list)
    recent_roundtrip_tool_summaries: list[RecentRoundtripToolSummary] = Field(default_factory=list)
    previous_user_request: str = ""
    latest_assistant_follow_up: str = ""


class ConversationMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sources: list[str] = Field(default_factory=list)
    source_conversation_id: UUID | None = None
    source_roundtrip_id: UUID | None = None
    source_message_index: int | None = None


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
    user_prompt: str
    generated_response: str
    roundtrip_summary: Optional[str]
    roundtrip_summary_embedding: Optional[list[float]]
    response_payload: OrchestratorPayload
    parsed_query: dict[str, Any]
    created_at: str
    resolved_model_config: ConversationModelConfig | None = None
    model: Optional[str] = None
    feedback_id: Optional[UUID] = None
    assistant_follow_up: str = ""
    relevant_evidence: RelevantEvidenceByTool = field(default_factory=RelevantEvidenceByTool.empty)

    def __post_init__(self) -> None:
        self.response_payload = OrchestratorPayload.model_validate(self.response_payload)
        self.parsed_query = dict(self.parsed_query) if isinstance(self.parsed_query, dict) else {}
        if self.resolved_model_config is not None:
            self.resolved_model_config = ConversationModelConfig.model_validate(self.resolved_model_config)
        self.relevant_evidence = RelevantEvidenceByTool.model_validate(self.relevant_evidence)


@dataclass(frozen=False)
class ConversationEvent:
    id: int
    conversation_id: UUID
    roundtrip_id: Optional[UUID]
    event_type: str
    source: str
    agent_name: str
    node_name: str
    iteration: Optional[int]
    payload: dict[str, Any]
    created_at: str


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
