from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from llm.clients.embeddings import embed_text
from personalization.user_attributes.models.user_attribute_models import UserAttributeSearchResult
from personalization.user_attributes.models.user_attribute_types import ATTRIBUTE_TYPE_COMPACT_DESCRIPTION, UserAttributeType
from personalization.user_attributes.repository.repo_factory import get_user_attribute_repo


class SearchUserAttributesArgs(BaseModel):
    query: str = Field(
        ...,
        description="Short literal search phrase for the attribute to find.",
    )
    limit: int = Field(default=5, ge=1, le=10, description="Maximum number of matching attributes to return.")
    is_active: bool | None = Field(default=True, description="Optional active-attribute filter.")
    attribute_type: UserAttributeType | None = Field(default=None, description=f"Optional attribute-type filter. {ATTRIBUTE_TYPE_COMPACT_DESCRIPTION}")
    group_key: str | None = Field(default=None, description="Optional grouping-key filter.")
    source: str | None = Field(default=None, description="Optional source filter.")


SEARCH_USER_ATTRIBUTES_DESCRIPTION = "Search persistent user attributes by semantic similarity."


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
    group_key: str | None = None,
    source: str | None = None,
) -> list[UserAttributeSearchResult]:
    query_embedding = embed_text(query)
    return get_user_attribute_repo().search_attributes(
        query_embedding=query_embedding,
        limit=limit,
        is_active=is_active,
        attribute_type=attribute_type,
        group_key=group_key,
        source=source,
    )
