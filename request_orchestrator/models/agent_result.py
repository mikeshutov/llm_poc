from __future__ import annotations

from dataclasses import dataclass, field

from request_orchestrator.models.evidence import ToolResult


@dataclass(frozen=True)
class AgentResult:
    tool_results: list[ToolResult] = field(default_factory=list)
    relevant_evidence_ids: list[str] = field(default_factory=list)

    def copy(
        self,
        *,
        tool_results: list[ToolResult] | None = None,
        relevant_evidence_ids: list[str] | None = None,
    ) -> "AgentResult":
        return AgentResult(
            tool_results=(
                [tool_result.model_copy(deep=True) for tool_result in self.tool_results]
                if tool_results is None
                else [tool_result.model_copy(deep=True) for tool_result in tool_results]
            ),
            relevant_evidence_ids=(
                list(self.relevant_evidence_ids)
                if relevant_evidence_ids is None
                else list(relevant_evidence_ids)
            ),
        )
