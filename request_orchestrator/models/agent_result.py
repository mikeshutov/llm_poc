from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from uuid import UUID


class AnswerStatus(StrEnum):
    PENDING = "pending"
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"


class ExecutionStatus(StrEnum):
    """Outcome of executing the planned work, independent of answerability."""

    PENDING = "pending"
    SUCCESS = "success"
    PARTIAL = "partial"
    NO_NEW_WORK = "no_new_work"
    FAILED = "failed"


# Compatibility alias for callers that still import the old name.
ResultStatus = AnswerStatus


@dataclass(frozen=True)
class AgentResult:
    tool_call_ids: list[UUID] = field(default_factory=list)
    relevant_evidence_ids: list[UUID] = field(default_factory=list)
    answer_status: AnswerStatus = AnswerStatus.PENDING
    execution_status: ExecutionStatus = ExecutionStatus.PENDING

    @property
    def result_status(self) -> AnswerStatus:
        """Backward-compatible name for answer_status."""
        return self.answer_status

    def with_recorded_tool_call(self, tool_call_id: UUID) -> "AgentResult":
        if tool_call_id in self.tool_call_ids:
            return self
        return self.copy(tool_call_ids=[*self.tool_call_ids, tool_call_id])

    def copy(
        self,
        *,
        tool_call_ids: list[UUID] | None = None,
        relevant_evidence_ids: list[UUID] | None = None,
        answer_status: AnswerStatus | None = None,
        execution_status: ExecutionStatus | None = None,
        result_status: AnswerStatus | None = None,
    ) -> "AgentResult":
        resolved_answer_status = (
            answer_status
            if answer_status is not None
            else result_status
            if result_status is not None
            else self.answer_status
        )
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
            answer_status=resolved_answer_status,
            execution_status=(
                self.execution_status
                if execution_status is None
                else execution_status
            ),
        )
