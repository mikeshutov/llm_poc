from __future__ import annotations

import pytest
from pydantic import ValidationError

from services.attribute_classifier.models import (
    AttributeCandidate,
    AttributeClassificationRequest,
)


def test_request_accepts_string_and_structured_attributes() -> None:
    request = AttributeClassificationRequest(
        context="winter hiking",
        attributes=["food.likes", {"id": "color", "text": "preferred color"}],
    )

    assert request.attributes[0] == "food.likes"
    assert request.attributes[1] == AttributeCandidate(id="color", text="preferred color")


def test_request_rejects_extra_fields_and_empty_values() -> None:
    with pytest.raises(ValidationError):
        AttributeClassificationRequest(
            context="winter hiking",
            attributes=[{"text": "color", "unexpected": True}],
        )
    with pytest.raises(ValidationError):
        AttributeClassificationRequest(context="", attributes=[])
