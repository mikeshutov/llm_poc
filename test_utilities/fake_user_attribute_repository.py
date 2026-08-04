from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

from personalization.user_attributes.models.user_attribute_models import UserAttribute


@dataclass
class FakeUserAttributeRepository:
    created_attributes: list[UserAttribute] = field(default_factory=list)

    def create_attribute(
        self,
        *,
        value: list[str],
        attribute_embedding: list[float] | None,
        attribute_type: str,
        source: str,
        confidence: float | None = None,
        importance: float | None = None,
    ) -> UserAttribute:
        attribute = UserAttribute(
            id=uuid4(),
            user_id=None,
            value=value,
            attribute_embedding=attribute_embedding,
            attribute_type=attribute_type,
            source=source,
            source_conversation_id=None,
            source_roundtrip_id=None,
            is_active=True,
            created_at='2026-08-04T00:00:00Z',
            updated_at='2026-08-04T00:00:00Z',
            confidence=confidence,
            importance=importance,
        )
        self.created_attributes.append(attribute)
        return attribute
