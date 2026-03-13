from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator


class PlanStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class PlanStep(BaseModel):
    id: str                                    # LLM reference e.g. "E1", used for arg substitution
    db_id: UUID = Field(default_factory=uuid4) # UUID for DB storage
    step_index: int = 0                        # assigned by Plan validator
    plan: str
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class Plan(BaseModel):
    steps: list[PlanStep]
    final_answer: str | None = None
    db_id: Optional[UUID] = None          # set after persisting to plans table
    current_step_index: int = 0           # tracks progress for failure recovery
    status: PlanStatus = PlanStatus.PENDING

    @model_validator(mode="after")
    def assign_step_indices(self) -> Plan:
        for i, step in enumerate(self.steps):
            step.step_index = i
        return self
