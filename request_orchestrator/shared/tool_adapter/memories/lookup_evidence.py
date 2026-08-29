from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from conversation.repository.repo_factory import get_conversation_repo
from request_orchestrator.models.evidence import HydratedEvidenceView, ToolResult
from request_orchestrator.shared.runtime_context import get_current_user_id
from tool.constants import TOOL_NAME_LOOKUP_EVIDENCE


class LookupEvidenceArgs(BaseModel):
    evidence_ids: list[str] = Field(
        ...,
        min_length=1,
        description="Evidence IDs from prior roundtrip context to retrieve. Provide all needed IDs in one array.",
    )


class EvidenceLookupResult(BaseModel):
    evidence: list[HydratedEvidenceView] = Field(default_factory=list)


@tool(
    TOOL_NAME_LOOKUP_EVIDENCE,
    args_schema=LookupEvidenceArgs,
    description="""
Retrieve full evidence records from the current user's prior conversation history.

Required fields:
- evidence_ids (array of strings): One or more evidence IDs from recent_roundtrips context.

Returns matching evidence in the same order as requested.
""",
)
def lookup_evidence(evidence_ids: list[str]) -> ToolResult:
    normalized_ids = list(
        dict.fromkeys(
            evidence_id.strip()
            for evidence_id in evidence_ids
            if isinstance(evidence_id, str) and evidence_id.strip()
        )
    )
    evidence_by_id = get_conversation_repo().get_evidence_by_ids_for_user(
        normalized_ids,
        user_id=get_current_user_id(),
    )
    evidence = [evidence_by_id[evidence_id] for evidence_id in normalized_ids if evidence_id in evidence_by_id]
    return ToolResult(
        result=EvidenceLookupResult(
            evidence=[evidence_view.hydrated_view() for evidence_view in evidence],
        ),
        evidence=evidence,
    )
