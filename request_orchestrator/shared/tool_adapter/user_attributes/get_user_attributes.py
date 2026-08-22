from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from common.data import normalize_string_list
from personalization.user_attributes.models.user_attribute_models import UserAttribute
from personalization.user_attributes.models.user_attribute_types import ATTRIBUTE_TYPE_COMPACT_DESCRIPTION, UserAttributeType
from personalization.user_attributes.repository.repo_factory import get_user_attribute_repo
from request_orchestrator.models.evidence import EvidenceView, HydratedEvidence, ToolResult
from request_orchestrator.shared.runtime_context import get_current_user_id
from tool.constants import TOOL_NAME_GET_USER_ATTRIBUTES
from tool.constants import TOOL_RESULT_TYPE_USER_ATTRIBUTE


class GetUserAttributesArgs(BaseModel):
    limit: int = Field(default=10, ge=1, le=50, description="Maximum number of attributes to return.")
    order_by: str = Field(default="updated_at", description="Sort field: created_at, updated_at, confidence, or importance.")
    descending: bool = Field(default=True, description="Whether to sort in descending order.")
    is_active: bool | None = Field(default=True, description="Optional active-attribute filter.")
    attribute_type: UserAttributeType | None = Field(default=None, description=f"Optional attribute-type filter. {ATTRIBUTE_TYPE_COMPACT_DESCRIPTION}")
    group_key: str | None = Field(default=None, description="Optional grouping-key filter.")
    source: str | None = Field(default=None, description="Optional source filter.")


GET_USER_ATTRIBUTES_DESCRIPTION = "List stored user attributes."


class UserAttributeMetadata(BaseModel):
    group_key: str | None = None
    source: str | None = None
    is_active: bool
    confidence: float | None = None
    importance: float | None = None
    created_at: str
    updated_at: str


def _attribute_summary(attribute: UserAttribute) -> str:
    return "; ".join(normalize_string_list(attribute.value)).strip() or "Stored user attribute."


def _tool_result(result: list[UserAttribute]) -> ToolResult:
    hydrated_evidence: list[HydratedEvidence] = []
    evidence_views: list[EvidenceView] = []
    for attribute in result:
        metadata = UserAttributeMetadata(
            group_key=attribute.group_key,
            source=attribute.source,
            is_active=attribute.is_active,
            confidence=attribute.confidence,
            importance=attribute.importance,
            created_at=attribute.created_at,
            updated_at=attribute.updated_at,
        )
        hydrated = HydratedEvidence(
            item_id=str(attribute.id),
            tool_name=TOOL_NAME_GET_USER_ATTRIBUTES,
            title=attribute.attribute_type,
            summary=_attribute_summary(attribute),
            published_at=attribute.updated_at,
            source=TOOL_NAME_GET_USER_ATTRIBUTES,
            entity_type=TOOL_RESULT_TYPE_USER_ATTRIBUTE,
            metadata=metadata.model_dump(exclude_none=True),
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
    TOOL_NAME_GET_USER_ATTRIBUTES,
    args_schema=GetUserAttributesArgs,
    description=GET_USER_ATTRIBUTES_DESCRIPTION,
)
def get_user_attributes(
    limit: int = 10,
    order_by: str = "updated_at",
    descending: bool = True,
    is_active: bool | None = True,
    attribute_type: UserAttributeType | None = None,
    group_key: str | None = None,
    source: str | None = None,
) -> ToolResult:
    return _tool_result(
        get_user_attribute_repo().list_attributes(
            limit=limit,
            order_by=order_by,
            descending=descending,
            user_id=get_current_user_id(),
            is_active=is_active,
            attribute_type=attribute_type,
            group_key=group_key,
            source=source,
        )
    )
