from __future__ import annotations

from pydantic import BaseModel, Field

from request_orchestrator.models.evidence import EvidenceView


class OrchestratorPayloadToolSummaryItem(BaseModel):
    entity_type: str = ""
    entity_id: str = ""


class OrchestratorPayloadToolSummary(BaseModel):
    evidence_produced: list[OrchestratorPayloadToolSummaryItem] = Field(default_factory=list)


class OrchestratorPayloadResultBlock(BaseModel):
    content: str = ""
    evidence_ids: list[str] = Field(default_factory=list)


class OrchestratorPayload(BaseModel):
    tool_results: list[dict] = Field(default_factory=list)
    relevant_evidence_ids: list[str] = Field(default_factory=list)
    result: list[OrchestratorPayloadResultBlock] = Field(default_factory=list)
    evidence_by_id: dict[str, EvidenceView] = Field(default_factory=dict)
    used_evidence_ids: list[str] = Field(default_factory=list)
    next_question: str = ""
    roundtrip_summary: str = ""
    tool_summary: OrchestratorPayloadToolSummary = Field(default_factory=OrchestratorPayloadToolSummary)
    roundtrip_latency_ms: int | None = None
