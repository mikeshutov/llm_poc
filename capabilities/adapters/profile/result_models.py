from __future__ import annotations

from pydantic import BaseModel, Field

from agent_runtime.models.agent_result import AnswerStatus, ExecutionStatus


class UserProfileUpdateResult(BaseModel):
    user_id: str | None
    first_name: str | None = None
    last_name: str | None = None
    display_name: str | None = None


class UserManagementResult(BaseModel):
    agent: str | None = None
    status: ExecutionStatus = ExecutionStatus.FAILED
    answer_status: AnswerStatus | None = None
    tool_names: list[str] = Field(default_factory=list)
    tool_call_ids: list[str] = Field(default_factory=list)
    error: str | None = None
    result_count: int = 0
