from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID


@dataclass(frozen=False)
class File:
    id: UUID
    file_path: str
    file_name: str
    file_type: str
    metadata: dict[str, Any]
    uploaded_at: str


@dataclass(frozen=True)
class FileChunkResult:
    file_id: UUID
    file_name: str
    file_path: str
    content: str
