from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from common.data import sanitize_for_json_storage
from request_orchestrator.models.evidence import HydratedEvidence
from request_orchestrator.models.synthesized_result import SynthesisResultBlock

if TYPE_CHECKING:
    from .agent_state import AgentState


@dataclass(frozen=True)
class AgentResult:
    answer: list[str]
    answer_blocks: list[SynthesisResultBlock] = field(default_factory=list)
    next_question: str = ""
    roundtrip_summary: str = ""
    roundtrip_latency_ms: int | None = None
    tool_summary: dict[str, Any] = field(default_factory=dict)
    agent_logs: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    used_evidence_ids: list[str] = field(default_factory=list)
    hydrated_evidence_by_id: dict[str, HydratedEvidence] = field(default_factory=dict)

    @property
    def raw_response(self) -> str:
        return "\n\n".join(p for p in self.answer if p)

    def to_payload_for_update_roundtrip(self) -> dict[str, Any]:
        payload = {
            "response": self.raw_response,
            "result": [block.model_dump() for block in self.answer_blocks],
            "used_evidence_ids": list(self.used_evidence_ids),
            "hydrated_evidence_by_id": {
                evidence_id: evidence.model_dump()
                for evidence_id, evidence in self.hydrated_evidence_by_id.items()
            },
            "next_question": self.next_question,
            "roundtrip_summary": self.roundtrip_summary,
            "tool_summary": self.tool_summary,
            "agent_logs": self.agent_logs,
        }
        if self.roundtrip_latency_ms is not None:
            payload["roundtrip_latency_ms"] = self.roundtrip_latency_ms
        return sanitize_for_json_storage(payload)

    @classmethod
    def from_state(
        cls,
        *,
        answer_blocks: list[SynthesisResultBlock],
        next_question: str | None = "",
        roundtrip_summary: str | None = "",
        tool_summary: dict[str, Any] | None = None,
        used_evidence_ids: list[str] | None = None,
        hydrated_evidence_by_id: dict[str, HydratedEvidence] | None = None,
        state: AgentState,
    ) -> "AgentResult":
        normalized_answer_blocks = [
            SynthesisResultBlock(
                content=block.content.strip(),
                evidence_ids=[evidence_id for evidence_id in block.evidence_ids if evidence_id],
            )
            for block in answer_blocks
            if block.content.strip()
        ]

        return cls(
            answer=[block.content for block in normalized_answer_blocks],
            answer_blocks=normalized_answer_blocks,
            next_question=(next_question or "").strip(),
            roundtrip_summary=roundtrip_summary or "",
            tool_summary=tool_summary or {},
            agent_logs=state.build_agent_logs(),
            used_evidence_ids=[] if used_evidence_ids is None else list(used_evidence_ids),
            hydrated_evidence_by_id={} if hydrated_evidence_by_id is None else dict(hydrated_evidence_by_id),
        )
