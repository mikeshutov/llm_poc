from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from uuid import UUID

class ResultStatus(StrEnum):
    PENDING = "pending"
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    NO_NEW_WORK = "no_new_work"
    FAILED = "failed"


@dataclass(frozen=True)
class AgentResult:
    tool_call_ids: list[UUID] = field(default_factory=list)
    relevant_evidence_ids: list[UUID] = field(default_factory=list)
    result_status: ResultStatus = ResultStatus.PENDING

    def with_recorded_tool_call(self, tool_call_id: UUID) -> "AgentResult":
        if tool_call_id in self.tool_call_ids:
            return self
        return self.copy(tool_call_ids=[*self.tool_call_ids, tool_call_id])

    def copy(
        self,
        *,
        tool_call_ids: list[UUID] | None = None,
        relevant_evidence_ids: list[UUID] | None = None,
        result_status: ResultStatus | None = None,
    ) -> "AgentResult":
        return AgentResult(
            tool_call_ids=(
                list(self.tool_call_ids)
                if tool_call_ids is None
                else list(tool_call_ids)
            ),
            relevant_evidence_ids=(
                list(self.relevant_evidence_ids)
                if relevant_evidence_ids is None
                else list(relevant_evidence_ids)
            ),
            result_status=self.result_status if result_status is None else result_status,
        )
