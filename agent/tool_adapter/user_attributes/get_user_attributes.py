from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from user_attributes.models.user_attribute_models import UserAttribute
from user_attributes.models.user_attribute_types import ATTRIBUTE_TYPE_DESCRIPTION, UserAttributeType
from user_attributes.repository.repo_factory import get_user_attribute_repo


class GetUserAttributesArgs(BaseModel):
    limit: int = Field(default=10, ge=1, le=50, description="Maximum number of attributes to return.")
    order_by: str = Field(default="updated_at", description="Sort field: created_at, updated_at, confidence, or importance.")
    descending: bool = Field(default=True, description="Whether to sort in descending order.")
    is_active: bool | None = Field(default=True, description="Optional active-attribute filter.")
    attribute_type: UserAttributeType | None = Field(default=None, description=f"Optional attribute type filter: {ATTRIBUTE_TYPE_DESCRIPTION}.")
    source: str | None = Field(default=None, description="Optional source filter.")


GET_USER_ATTRIBUTES_DESCRIPTION = f"""
List stored user attributes with ordering and optional filters.

Optional fields:
- limit (integer): Maximum number of attributes to return.
- order_by (string): created_at, updated_at, confidence, or importance.
- descending (boolean): Sort descending when true.
- is_active (boolean): Optional active-attribute filter.
- attribute_type (string): {ATTRIBUTE_TYPE_DESCRIPTION}.
- source (string): Optional source filter.
"""


@tool(
    "get_user_attributes",
    args_schema=GetUserAttributesArgs,
    description=GET_USER_ATTRIBUTES_DESCRIPTION,
)
def get_user_attributes(
    limit: int = 10,
    order_by: str = "updated_at",
    descending: bool = True,
    is_active: bool | None = True,
    attribute_type: UserAttributeType | None = None,
    source: str | None = None,
) -> list[UserAttribute]:
    return get_user_attribute_repo().list_attributes(
        limit=limit,
        order_by=order_by,
        descending=descending,
        is_active=is_active,
        attribute_type=attribute_type,
        source=source,
    )
