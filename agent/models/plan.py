from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class PlanStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class PlanStep(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    step_index: int
    plan: str
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class Plan(BaseModel):
    steps: list[PlanStep]
    final_answer: str | None = None
    db_id: Optional[UUID] = None          # set after persisting to plans table
    current_step_index: int = 0           # tracks progress for failure recovery
    status: PlanStatus = PlanStatus.PENDING
