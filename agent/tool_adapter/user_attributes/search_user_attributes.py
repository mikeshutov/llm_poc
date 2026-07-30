from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from llm.clients.embeddings import embed_text
from user_attributes.models.user_attribute_models import UserAttributeSearchResult
from user_attributes.models.user_attribute_types import ATTRIBUTE_TYPE_DESCRIPTION, UserAttributeType
from user_attributes.repository.repo_factory import get_user_attribute_repo


class SearchUserAttributesArgs(BaseModel):
    query: str = Field(..., description="Natural-language description of the user attribute to find.")
    limit: int = Field(default=5, ge=1, le=10, description="Maximum number of matching attributes to return.")
    is_active: bool | None = Field(default=True, description="Optional active-attribute filter.")
    attribute_type: UserAttributeType | None = Field(default=None, description=f"Optional attribute type filter: {ATTRIBUTE_TYPE_DESCRIPTION}.")
    source: str | None = Field(default=None, description="Optional source filter.")


SEARCH_USER_ATTRIBUTES_DESCRIPTION = f"""
Search persistent user attributes by semantic similarity.

Required fields:
- query (string): Natural-language description of the attribute to find.

Optional fields:
- limit (integer): Maximum number of matches to return. Defaults to 5.
- is_active (boolean): Optional active-attribute filter.
- attribute_type (string): {ATTRIBUTE_TYPE_DESCRIPTION}.
- source (string): Optional source filter.
"""


@tool(
    "search_user_attributes",
    args_schema=SearchUserAttributesArgs,
    description=SEARCH_USER_ATTRIBUTES_DESCRIPTION,
)
def search_user_attributes(
    query: str,
    limit: int = 5,
    is_active: bool | None = True,
    attribute_type: UserAttributeType | None = None,
    source: str | None = None,
) -> list[UserAttributeSearchResult]:
    query_embedding = embed_text(query)
    return get_user_attribute_repo().search_attributes(
        query_embedding=query_embedding,
        limit=limit,
        is_active=is_active,
        attribute_type=attribute_type,
        source=source,
    )
