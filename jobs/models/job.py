from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from request_orchestrator.models.plan import Plan
from jobs.models.schedule import JobSchedule


class JobPlanStatus(StrEnum):
    READY = "ready"
    STALE = "stale"
    FAILED = "failed"


class Job(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID
    user_id: str
    name: str
    prompt: str
    plan: Plan
    plan_version: int = Field(ge=1)
    plan_generated_at: datetime
    plan_status: JobPlanStatus = JobPlanStatus.READY
    plan_prompt_hash: str
    enabled: bool = False
    schedule: JobSchedule
    next_execution_at: datetime
    created_at: datetime
    updated_at: datetime

    @field_validator("user_id", "name", "prompt")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("value is required")
        return normalized
