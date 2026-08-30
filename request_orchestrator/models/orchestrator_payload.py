from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, RootModel

from request_orchestrator.models.evidence import EvidenceView


class EvidenceProducedByTool(RootModel[dict[str, list[UUID]]]):
    @classmethod
    def empty(cls) -> "EvidenceProducedByTool":
        return cls({})


class OrchestratorPayloadToolSummary(BaseModel):
    evidence_produced: EvidenceProducedByTool = Field(default_factory=EvidenceProducedByTool.empty)

    @classmethod
    def build(
        cls,
        evidence_by_id: dict[str, EvidenceView],
    ) -> "OrchestratorPayloadToolSummary":
        evidence_ids_by_tool: dict[str, list[UUID]] = {}
        seen_evidence_ids: set[UUID] = set()

        for evidence in evidence_by_id.values():
            tool_name = evidence.tool_name.strip()
            if not tool_name or evidence.id in seen_evidence_ids:
                continue
            seen_evidence_ids.add(evidence.id)
            evidence_ids_by_tool.setdefault(tool_name, []).append(evidence.id)

        return cls(evidence_produced=EvidenceProducedByTool(evidence_ids_by_tool))


class OrchestratorPayloadResultBlock(BaseModel):
    content: str = ""
    evidence_ids: list[str] = Field(default_factory=list)


class OrchestratorPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result: list[OrchestratorPayloadResultBlock] = Field(default_factory=list)
    evidence_by_id: dict[str, EvidenceView] = Field(default_factory=dict)
    used_evidence_ids: list[str] = Field(default_factory=list)
    next_question: str = ""
    roundtrip_summary: str = ""
    tool_summary: OrchestratorPayloadToolSummary = Field(default_factory=OrchestratorPayloadToolSummary)
    roundtrip_latency_ms: int | None = None
