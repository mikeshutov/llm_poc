from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel


class ToolCall(BaseModel):
    id: UUID
    roundtrip_id: UUID
    plan_id: Optional[UUID] = None
    plan_step_id: Optional[UUID] = None
    step_index: Optional[int] = None
    call_index: int
    tool_name: str
    status: str
    input_payload: dict[str, Any] = {}
    output_payload: Optional[dict[str, Any]] = None
    error_message: Optional[str] = None
    duration_ms: Optional[int] = None
    goal: Optional[str] = None
    summary: Optional[str] = None
    created_at: datetime
