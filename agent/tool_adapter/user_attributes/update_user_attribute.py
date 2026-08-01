from __future__ import annotations

from uuid import UUID

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from common.parsing import normalize_string_list
from llm.clients.embeddings import embed_text
from user_attributes.models.user_attribute_models import UserAttribute
from user_attributes.models.user_attribute_types import ATTRIBUTE_TYPE_DESCRIPTION, UserAttributeType
from user_attributes.repository.repo_factory import get_user_attribute_repo


class UpdateUserAttributeArgs(BaseModel):
    attribute_id: str = Field(..., description="The id of the attribute to update.")
    value: list[str] | None = Field(default=None, description="Updated attribute values as an array/list of strings. Pass a JSON array, not a single string.")
    attribute_type: UserAttributeType = Field(..., description=f"Updated attribute type: {ATTRIBUTE_TYPE_DESCRIPTION}.")
    source: str | None = Field(default=None, description="Updated source value such as explicit, derived, or computed.")
    is_active: bool | None = Field(default=None, description="Whether the attribute should remain active.")
    confidence: float | None = Field(default=None, ge=0, le=1, description="Optional confidence score between 0 and 1.")
    importance: float | None = Field(default=None, ge=0, le=1, description="Optional importance score between 0 and 1.")


UPDATE_USER_ATTRIBUTE_DESCRIPTION = f"""
Update an existing persistent user attribute.

Required fields:
- attribute_id (string): The attribute id to update.
- attribute_type (string): {ATTRIBUTE_TYPE_DESCRIPTION}.

Optional fields:
- value (array/list of strings): Updated attribute values. Pass a JSON array, not a single string. Store only concrete user-specific values such as ["pizza", "eggs"] or ["Python", "React"], not labels, summaries, placeholders, or brace-wrapped text like `{"dietary staples mentioned by the user"}`.
- source (string): Updated source value.
- is_active (boolean): Set false to deactivate an attribute.
- confidence (number): Optional 0..1 confidence score.
- importance (number): Optional 0..1 importance score.
"""


def _value_text(value: list[str]) -> str:
    return "; ".join(normalize_string_list(value))


@tool(
    "update_user_attribute",
    args_schema=UpdateUserAttributeArgs,
    description=UPDATE_USER_ATTRIBUTE_DESCRIPTION,
)
def update_user_attribute(
    attribute_id: str,
    attribute_type: UserAttributeType,
    value: list[str] | None = None,
    source: str | None = None,
    is_active: bool | None = None,
    confidence: float | None = None,
    importance: float | None = None,
) -> UserAttribute | None:
    try:
        parsed_attribute_id = UUID(attribute_id)
    except ValueError:
        return None

    attribute_embedding = embed_text(_value_text(value)) if value else None
    return get_user_attribute_repo().update_attribute(
        attribute_id=parsed_attribute_id,
        value=value,
        attribute_embedding=attribute_embedding,
        attribute_type=attribute_type,
        source=source,
        is_active=is_active,
        confidence=confidence,
        importance=importance,
    )
