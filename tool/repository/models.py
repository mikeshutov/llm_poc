from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    id: UUID
    roundtrip_id: UUID
    plan_id: Optional[UUID] = None
    plan_step_id: Optional[UUID] = None
    step_index: Optional[int] = None
    tool_name: str
    request_hash: str
    status: str
    input_payload: dict[str, Any] = Field(default_factory=dict)
    output_payload: Any | None = None
    error_message: Optional[str] = None
    duration_ms: Optional[int] = None
    goal: Optional[str] = None
    hypothesis: str = ""
    call_metadata: dict[str, Any] = Field(default_factory=dict)
    summary: Optional[str] = None
    created_at: datetime
