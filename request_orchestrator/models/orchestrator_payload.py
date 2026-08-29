from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field, RootModel

from request_orchestrator.models.evidence import EvidenceView


class EvidenceProducedByTool(RootModel[dict[str, list[UUID]]]):
    @classmethod
    def empty(cls) -> "EvidenceProducedByTool":
        return cls({})


class OrchestratorPayloadToolSummary(BaseModel):
    evidence_produced: EvidenceProducedByTool = Field(default_factory=EvidenceProducedByTool.empty)


class OrchestratorPayloadResultBlock(BaseModel):
    content: str = ""
    evidence_ids: list[str] = Field(default_factory=list)


class OrchestratorPayload(BaseModel):
    tool_results: list[dict] = Field(default_factory=list)
    result: list[OrchestratorPayloadResultBlock] = Field(default_factory=list)
    evidence_by_id: dict[str, EvidenceView] = Field(default_factory=dict)
    used_evidence_ids: list[str] = Field(default_factory=list)
    next_question: str = ""
    roundtrip_summary: str = ""
    tool_summary: OrchestratorPayloadToolSummary = Field(default_factory=OrchestratorPayloadToolSummary)
    roundtrip_latency_ms: int | None = None
