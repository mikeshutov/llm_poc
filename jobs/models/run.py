from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class JobRunStatus(StrEnum):
    PENDING = "pending"
    ENQUEUED = "enqueued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class JobRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID
    job_id: UUID
    scheduled_for: datetime
    status: JobRunStatus = JobRunStatus.PENDING
    queue_message_id: str | None = None
    deduplication_key: str
    created_at: datetime
    updated_at: datetime
