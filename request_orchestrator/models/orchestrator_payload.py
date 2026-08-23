from __future__ import annotations

from pydantic import BaseModel, Field


class OrchestratorPayloadResultBlock(BaseModel):
    content: str = ""
    evidence_ids: list[str] = Field(default_factory=list)


class OrchestratorPayload(BaseModel):
    tool_results: list[dict] = Field(default_factory=list)
    relevant_evidence_ids: list[str] = Field(default_factory=list)
    result: list[OrchestratorPayloadResultBlock] = Field(default_factory=list)
    evidence_by_id: dict[str, dict] = Field(default_factory=dict)
    used_evidence_ids: list[str] = Field(default_factory=list)
    next_question: str = ""
    roundtrip_summary: str = ""
    roundtrip_latency_ms: int | None = None
