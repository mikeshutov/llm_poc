from __future__ import annotations

from pydantic import BaseModel


class GetFileByIdResult(BaseModel):
    file_id: str | None = None
    file_name: str | None = None
    file_path: str | None = None
    file_type: str | None = None
    uploaded_at: str | None = None
    first_chunk: str | None = None
    error: str | None = None
