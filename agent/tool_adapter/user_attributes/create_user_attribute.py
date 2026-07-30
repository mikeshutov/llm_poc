from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from llm.clients.embeddings import embed_text
from user_attributes.models.user_attribute_models import UserAttribute
from user_attributes.models.user_attribute_types import ATTRIBUTE_TYPE_DESCRIPTION, UserAttributeType
from user_attributes.repository.repo_factory import get_user_attribute_repo


class CreateUserAttributeArgs(BaseModel):
    attribute_text: str = Field(..., description="The user attribute text to store.")
    attribute_type: UserAttributeType = Field(..., description=f"Attribute type: {ATTRIBUTE_TYPE_DESCRIPTION}.")
    source: str = Field(default="explicit", description="Attribute source such as explicit, derived, or computed.")
    confidence: float | None = Field(default=None, ge=0, le=1, description="Optional confidence score between 0 and 1.")
    importance: float | None = Field(default=None, ge=0, le=1, description="Optional importance score between 0 and 1.")


CREATE_USER_ATTRIBUTE_DESCRIPTION = f"""
Create a persistent user attribute for future profile assembly and recall.

Required fields:
- attribute_text (string): The stable characteristic, preference, fact, or instruction to store.
- attribute_type (string): {ATTRIBUTE_TYPE_DESCRIPTION}.

Optional fields:
- source (string): explicit, derived, or computed. Defaults to explicit.
- confidence (number): Optional 0..1 confidence score.
- importance (number): Optional 0..1 importance score.
"""


@tool(
    "create_user_attribute",
    args_schema=CreateUserAttributeArgs,
    description=CREATE_USER_ATTRIBUTE_DESCRIPTION,
)
def create_user_attribute(
    attribute_text: str,
    attribute_type: UserAttributeType,
    source: str = "explicit",
    confidence: float | None = None,
    importance: float | None = None,
) -> UserAttribute:
    return get_user_attribute_repo().create_attribute(
        attribute_text=attribute_text,
        attribute_embedding=embed_text(attribute_text),
        attribute_type=attribute_type,
        source=source,
        confidence=confidence,
        importance=importance,
    )
