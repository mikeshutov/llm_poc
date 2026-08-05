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

    def list_attributes(
        self,
        *,
        limit: int = 50,
        order_by: str = 'updated_at',
        descending: bool = True,
        user_id: str | None = None,
        is_active: bool | None = None,
        attribute_type: str | None = None,
        source: str | None = None,
        source_conversation_id=None,
        source_roundtrip_id=None,
    ) -> list[UserAttribute]:
        attributes = list(self.created_attributes)

        if is_active is not None:
            attributes = [attribute for attribute in attributes if attribute.is_active == is_active]
        if attribute_type is not None:
            attributes = [attribute for attribute in attributes if attribute.attribute_type == attribute_type]
        if source is not None:
            attributes = [attribute for attribute in attributes if attribute.source == source]

        reverse = descending
        if order_by == 'created_at':
            attributes.sort(key=lambda attribute: attribute.created_at, reverse=reverse)
        elif order_by == 'updated_at':
            attributes.sort(key=lambda attribute: attribute.updated_at, reverse=reverse)
        elif order_by == 'confidence':
            attributes.sort(key=lambda attribute: attribute.confidence or 0.0, reverse=reverse)
        elif order_by == 'importance':
            attributes.sort(key=lambda attribute: attribute.importance or 0.0, reverse=reverse)

        return attributes[:limit]
