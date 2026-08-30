from __future__ import annotations

from dataclasses import dataclass, field

from request_orchestrator.models.agent_result import AgentResult
from request_orchestrator.models.evidence import EvidenceView
from request_orchestrator.models.orchestrator_payload import (
    OrchestratorPayload,
    OrchestratorPayloadResultBlock,
    OrchestratorPayloadToolSummary,
)
from request_orchestrator.models.synthesized_result import SynthesisResultBlock
from request_orchestrator.models.relevant_evidence import RelevantEvidenceByTool
from request_orchestrator.shared.evidence import build_evidence_bundle_from_tool_results
from tool.constants import EVIDENCE_PERSISTENCE_EXCLUDED_TOOL_NAMES


def _normalize_evidence_ids(
    evidence_ids: list[str],
    evidence_by_id: dict[str, EvidenceView],
) -> list[str]:
    if not evidence_ids or not evidence_by_id:
        return []

    normalized_ids: list[str] = []
    for evidence_id in evidence_ids:
        if not isinstance(evidence_id, str):
            continue
        normalized = evidence_id.strip()
        if not normalized:
            continue
        if normalized in evidence_by_id:
            normalized_ids.append(normalized)
    return normalized_ids


def _is_evidence_excluded_from_persistence(evidence: EvidenceView) -> bool:
    return evidence.tool_name.strip() in EVIDENCE_PERSISTENCE_EXCLUDED_TOOL_NAMES


@dataclass(frozen=True)
class OrchestratorResult:
    agent_result: AgentResult = field(default_factory=AgentResult)
    result_blocks: list[SynthesisResultBlock] = field(default_factory=list)
    answer: list[str] = field(default_factory=list)
    next_question: str = ""
    roundtrip_summary: str = ""
    roundtrip_latency_ms: int | None = None

    @property
    def raw_response(self) -> str:
        paragraphs = self.answer
        if self.result_blocks:
            paragraphs = [block.content for block in self.result_blocks if block.content]
        return "\n\n".join(paragraph for paragraph in paragraphs if paragraph)

    @property
    def used_evidence_ids(self) -> list[str]:
        used_evidence_ids: list[str] = []
        seen_evidence_ids: set[str] = set()
        for block in self.result_blocks:
            for evidence_id in block.evidence_ids:
                normalized = evidence_id.strip()
                if not normalized or normalized in seen_evidence_ids:
                    continue
                seen_evidence_ids.add(normalized)
                used_evidence_ids.append(normalized)
        return used_evidence_ids

    def with_roundtrip_latency(self, roundtrip_latency_ms: int | None) -> "OrchestratorResult":
        return OrchestratorResult(
            agent_result=self.agent_result,
            result_blocks=[block.model_copy(deep=True) for block in self.result_blocks],
            answer=list(self.answer),
            next_question=self.next_question,
            roundtrip_summary=self.roundtrip_summary,
            roundtrip_latency_ms=roundtrip_latency_ms,
        )

    def copy(
        self,
        *,
        agent_result: AgentResult | None = None,
        result_blocks: list[SynthesisResultBlock] | None = None,
        answer: list[str] | None = None,
        next_question: str | None = None,
        roundtrip_summary: str | None = None,
    ) -> "OrchestratorResult":
        return OrchestratorResult(
            agent_result=self.agent_result.copy() if agent_result is None else agent_result.copy(),
            result_blocks=(
                [block.model_copy(deep=True) for block in self.result_blocks]
                if result_blocks is None
                else [block.model_copy(deep=True) for block in result_blocks]
            ),
            answer=list(self.answer) if answer is None else list(answer),
            next_question=self.next_question if next_question is None else next_question,
            roundtrip_summary=self.roundtrip_summary if roundtrip_summary is None else roundtrip_summary,
            roundtrip_latency_ms=self.roundtrip_latency_ms,
        )

    def to_persistence_models(self) -> tuple[OrchestratorPayload, RelevantEvidenceByTool]:
        from tool.repository.tool_call_repository import ToolCallRepository

        tool_results = ToolCallRepository().get_tool_results(self.agent_result.tool_call_ids)
        evidence_bundle = build_evidence_bundle_from_tool_results(tool_results)
        evidence_by_id = {
            evidence_id: evidence
            for evidence_id, evidence in evidence_bundle.evidence_by_id.items()
            if not _is_evidence_excluded_from_persistence(evidence)
        }
        result_blocks = self.result_blocks
        if not result_blocks:
            result_blocks = [
                SynthesisResultBlock(content=paragraph.strip(), evidence_ids=[])
                for paragraph in self.answer
                if isinstance(paragraph, str) and paragraph.strip()
            ]
        normalized_result_blocks = [
            SynthesisResultBlock(
                content=block.content,
                evidence_ids=_normalize_evidence_ids(
                    block.evidence_ids,
                    evidence_by_id,
                ),
            )
            for block in result_blocks
        ]
        payload = OrchestratorPayload(
            result=[
                OrchestratorPayloadResultBlock(
                    content=block.content,
                    evidence_ids=list(block.evidence_ids),
                )
                for block in normalized_result_blocks
            ],
            evidence_by_id=evidence_by_id,
            used_evidence_ids=_normalize_evidence_ids(
                self.used_evidence_ids,
                evidence_by_id,
            ),
            next_question=self.next_question,
            roundtrip_summary=self.roundtrip_summary,
            tool_summary=OrchestratorPayloadToolSummary.build(evidence_by_id),
            roundtrip_latency_ms=self.roundtrip_latency_ms,
        )
        return (
            payload,
            RelevantEvidenceByTool.build(
                self.agent_result.relevant_evidence_ids,
                evidence_by_id,
            ),
        )
