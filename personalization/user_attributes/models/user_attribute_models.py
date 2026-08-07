from dataclasses import dataclass
from typing import Optional
from uuid import UUID


@dataclass(frozen=False)
class UserAttribute:
    id: UUID
    user_id: Optional[str]
    value: list[str]
    attribute_embedding: Optional[list[float]]
    attribute_type: str
    group_key: Optional[str]
    source: Optional[str]
    is_active: bool
    created_at: str
    updated_at: str
    confidence: Optional[float]
    importance: Optional[float]


@dataclass(frozen=False)
class UserAttributeSearchResult:
    id: UUID
    user_id: Optional[str]
    value: list[str]
    attribute_type: str
    group_key: Optional[str]
    source: Optional[str]
    is_active: bool
    created_at: str
    updated_at: str
    confidence: Optional[float]
    importance: Optional[float]
    relevance_score: float
