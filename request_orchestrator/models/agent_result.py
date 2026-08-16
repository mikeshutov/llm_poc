from __future__ import annotations

from dataclasses import dataclass, field

from request_orchestrator.models.evidence import ToolResult


@dataclass(frozen=True)
class AgentResult:
    tool_results: list[ToolResult] = field(default_factory=list)
    relevant_evidence_ids: list[str] = field(default_factory=list)

    def tool_results_by_step_id(self) -> dict[str, ToolResult]:
        return {
            tool_result.step_id: tool_result
            for tool_result in self.tool_results
            if tool_result.step_id.strip()
        }

    def with_recorded_tool_result(self, tool_result: ToolResult) -> "AgentResult":
        resolved_tool_result = tool_result.model_copy(deep=True)
        updated_tool_results = [existing.model_copy(deep=True) for existing in self.tool_results]
        if resolved_tool_result.step_id.strip():
            for index, existing in enumerate(updated_tool_results):
                if existing.step_id.strip() == resolved_tool_result.step_id.strip():
                    updated_tool_results[index] = resolved_tool_result
                    return self.copy(tool_results=updated_tool_results)
        updated_tool_results.append(resolved_tool_result)
        return self.copy(tool_results=updated_tool_results)

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
