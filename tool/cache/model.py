from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class ToolCache(BaseModel):
    id: UUID | None = None
    cache_key: str
    tool_name: str
    tool_version: int
    input_payload: dict[str, Any] = Field(default_factory=dict)
    output_payload: Any
    expires_at: datetime
    created_at: datetime | None = None
    updated_at: datetime | None = None
