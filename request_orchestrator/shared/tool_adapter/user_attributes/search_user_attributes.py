from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from common.data import normalize_string_list
from llm.clients.embeddings import embed_text
from personalization.user_attributes.models.user_attribute_models import UserAttributeSearchResult
from personalization.user_attributes.models.user_attribute_types import ATTRIBUTE_TYPE_COMPACT_DESCRIPTION, UserAttributeType
from personalization.user_attributes.repository.repo_factory import get_user_attribute_repo
from request_orchestrator.models.evidence import EvidenceView, HydratedEvidence, ToolResult
from request_orchestrator.shared.runtime_context import get_current_user_id
from tool.constants import TOOL_NAME_SEARCH_USER_ATTRIBUTES
from tool.constants import TOOL_RESULT_TYPE_USER_ATTRIBUTE


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


def _attribute_summary(attribute: UserAttributeSearchResult) -> str:
    return "; ".join(normalize_string_list(attribute.value)).strip() or "Matched user attribute."


def _tool_result(result: list[UserAttributeSearchResult]) -> ToolResult:
    hydrated_evidence: list[HydratedEvidence] = []
    evidence_views: list[EvidenceView] = []
    for attribute in result:
        hydrated = HydratedEvidence(
            item_id=str(attribute.id),
            tool_name=TOOL_NAME_SEARCH_USER_ATTRIBUTES,
            title=attribute.attribute_type,
            summary=_attribute_summary(attribute),
            published_at=attribute.updated_at,
            source=TOOL_NAME_SEARCH_USER_ATTRIBUTES,
            entity_type=TOOL_RESULT_TYPE_USER_ATTRIBUTE,
            metadata={
                "group_key": attribute.group_key,
                "source": attribute.source,
                "is_active": attribute.is_active,
                "confidence": attribute.confidence,
                "importance": attribute.importance,
                "created_at": attribute.created_at,
                "updated_at": attribute.updated_at,
                "relevance_score": attribute.relevance_score,
            },
            raw_payload=attribute,
        )
        hydrated_evidence.append(hydrated)
        evidence_views.append(
            EvidenceView(
                item_id=hydrated.item_id,
                title=hydrated.title,
                summary=hydrated.summary,
                metadata=dict(hydrated.metadata),
            )
        )
    return ToolResult(result=result, evidence_views=evidence_views, hydrated_evidence=hydrated_evidence)


@tool(
    TOOL_NAME_SEARCH_USER_ATTRIBUTES,
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
) -> ToolResult:
    query_embedding = embed_text(query)
    return _tool_result(
        get_user_attribute_repo().search_attributes(
            query_embedding=query_embedding,
            limit=limit,
            user_id=get_current_user_id(),
            is_active=is_active,
            attribute_type=attribute_type,
            group_key=group_key,
            source=source,
        )
    )
