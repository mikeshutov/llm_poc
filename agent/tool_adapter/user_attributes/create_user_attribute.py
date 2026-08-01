from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from llm.clients.embeddings import embed_text
from user_attributes.models.user_attribute_models import UserAttribute
from user_attributes.models.user_attribute_types import ATTRIBUTE_TYPE_DESCRIPTION, UserAttributeType
from user_attributes.repository.repo_factory import get_user_attribute_repo


class CreateUserAttributeArgs(BaseModel):
    value: list[str] = Field(..., description="The user attribute values to store as an array/list of strings. Pass a JSON array, not a single string.")
    attribute_type: UserAttributeType = Field(..., description=f"Attribute type: {ATTRIBUTE_TYPE_DESCRIPTION}.")
    source: str = Field(default="explicit", description="Attribute source such as explicit, derived, or computed.")
    confidence: float | None = Field(default=None, ge=0, le=1, description="Optional confidence score between 0 and 1.")
    importance: float | None = Field(default=None, ge=0, le=1, description="Optional importance score between 0 and 1.")


CREATE_USER_ATTRIBUTE_DESCRIPTION = f"""
Create a persistent user attribute for future profile assembly and recall.

Required fields:
- value (array/list of strings): The stable characteristic, preference, fact, or instruction values to store. Pass a JSON array, not a single string. Store only concrete user-specific values such as ["pizza", "eggs"] or ["Python", "React"], not labels, summaries, placeholders, or brace-wrapped text like `{"dietary staples mentioned by the user"}`.
- attribute_type (string): {ATTRIBUTE_TYPE_DESCRIPTION}.

Optional fields:
- source (string): explicit, derived, or computed. Defaults to explicit.
- confidence (number): Optional 0..1 confidence score.
- importance (number): Optional 0..1 importance score.
"""


def _value_text(value: list[str]) -> str:
    normalized = [value.strip() for value in value if value and value.strip()]
    if not normalized:
        raise ValueError("value must contain at least one non-empty string.")
    return "; ".join(normalized)


@tool(
    "create_user_attribute",
    args_schema=CreateUserAttributeArgs,
    description=CREATE_USER_ATTRIBUTE_DESCRIPTION,
)
def create_user_attribute(
    value: list[str],
    attribute_type: UserAttributeType,
    source: str = "explicit",
    confidence: float | None = None,
    importance: float | None = None,
) -> UserAttribute:
    return get_user_attribute_repo().create_attribute(
        value=value,
        attribute_embedding=embed_text(_value_text(value)),
        attribute_type=attribute_type,
        source=source,
        confidence=confidence,
        importance=importance,
    )
