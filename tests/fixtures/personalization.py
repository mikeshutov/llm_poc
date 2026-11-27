from copy import deepcopy
from uuid import uuid4

import pytest

from personalization.user_attributes.models.user_attribute_models import UserAttribute


@pytest.fixture
def user_attribute_factory():
    def create(**overrides):
        return UserAttribute(**deepcopy({
            "id": uuid4(), "user_id": None, "value": ["pizza"],
            "attribute_embedding": None, "attribute_type": "food.preference",
            "group_key": None, "source": "explicit", "is_active": True,
            "confidence": None, "importance": None,
            "created_at": "2026-08-05T00:00:00Z", "updated_at": "2026-08-05T00:00:00Z",
            **overrides,
        }))

    return create
