from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from tool.models.tool_definition import RateLimitPolicy, RetryPolicy


class RegisteredTool(BaseModel):
    id: UUID
    name: str
    version: int = 1
    description: str = ""
    result_type: str = "generic"
    rate_limit_key: str | None = None
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    rate_limit_policy: RateLimitPolicy | None = None
    is_active: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @field_validator("name", "description", "result_type", "rate_limit_key", mode="before")
    @classmethod
    def normalize_text(cls, value: str | None) -> str | None:
        return value.strip() if isinstance(value, str) else value

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: int) -> int:
        if value < 1:
            raise ValueError("version must be positive")
        return value

