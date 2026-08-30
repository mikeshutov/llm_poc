from uuid import UUID

from pydantic import RootModel

from request_orchestrator.models.evidence import EvidenceView


class RelevantEvidenceByTool(RootModel[dict[str, list[UUID]]]):
    """Evaluator-selected evidence IDs grouped by the tool that produced them."""

    @classmethod
    def empty(cls) -> "RelevantEvidenceByTool":
        return cls({})

    @classmethod
    def build(
        cls,
        relevant_evidence_ids: list[UUID],
        evidence_by_id: dict[str, EvidenceView],
    ) -> "RelevantEvidenceByTool":
        evidence_ids_by_tool: dict[str, list[UUID]] = {}
        seen_evidence_ids: set[UUID] = set()

        for evidence_id in relevant_evidence_ids:
            if evidence_id in seen_evidence_ids:
                continue
            seen_evidence_ids.add(evidence_id)
            evidence = evidence_by_id.get(str(evidence_id))
            tool_name = evidence.tool_name.strip() if evidence is not None else ""
            if tool_name:
                evidence_ids_by_tool.setdefault(tool_name, []).append(evidence_id)

        return cls(evidence_ids_by_tool)
