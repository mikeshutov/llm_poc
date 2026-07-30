from dataclasses import dataclass
from typing import Optional
from uuid import UUID


@dataclass(frozen=False)
class UserAttribute:
    id: UUID
    user_id: Optional[str]
    attribute_text: str
    attribute_embedding: Optional[list[float]]
    attribute_type: Optional[str]
    source: Optional[str]
    source_conversation_id: Optional[UUID]
    source_roundtrip_id: Optional[UUID]
    is_active: bool
    created_at: str
    updated_at: str
    confidence: Optional[float]
    importance: Optional[float]


@dataclass(frozen=False)
class UserAttributeSearchResult:
    id: UUID
    user_id: Optional[str]
    attribute_text: str
    attribute_type: Optional[str]
    source: Optional[str]
    source_conversation_id: Optional[UUID]
    source_roundtrip_id: Optional[UUID]
    is_active: bool
    created_at: str
    updated_at: str
    confidence: Optional[float]
    importance: Optional[float]
    relevance_score: float
