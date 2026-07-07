from dataclasses import dataclass
from typing import Optional
from uuid import UUID


@dataclass(frozen=False)
class Memory:
    id: UUID
    user_id: Optional[str]
    memory_text: str
    memory_embedding: Optional[list[float]]
    memory_type: Optional[str]
    source: Optional[str]
    source_conversation_id: Optional[UUID]
    source_roundtrip_id: Optional[UUID]
    is_active: bool
    created_at: str
    updated_at: str
    confidence: Optional[float]
    importance: Optional[float]


@dataclass(frozen=False)
class MemorySearchResult:
    id: UUID
    user_id: Optional[str]
    memory_text: str
    memory_type: Optional[str]
    source: Optional[str]
    source_conversation_id: Optional[UUID]
    source_roundtrip_id: Optional[UUID]
    is_active: bool
    created_at: str
    updated_at: str
    confidence: Optional[float]
    importance: Optional[float]
    relevance_score: float
