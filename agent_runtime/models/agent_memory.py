from __future__ import annotations

from pydantic import BaseModel


class AgentMemoryResponse(BaseModel):
    """Stable state handoff returned between delegated agents."""

    filename: str = ""
    file_type: str = "working_memory_file"
    contents: str = ""
    response: str = ""
    status: str = "pending"
    error: str | None = None
