from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Artifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID
    user_id: str
    artifact_type: str
    artifact_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime

    @field_validator("artifact_type", "artifact_id")
    @classmethod
    def validate_reference(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("artifact_type and artifact_id are required")
        return normalized
