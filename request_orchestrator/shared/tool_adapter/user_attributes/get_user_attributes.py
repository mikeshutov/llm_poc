from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from personalization.user_attributes.models.user_attribute_models import UserAttribute
from personalization.user_attributes.models.user_attribute_types import ATTRIBUTE_TYPE_COMPACT_DESCRIPTION, UserAttributeType
from personalization.user_attributes.repository.repo_factory import get_user_attribute_repo
from request_orchestrator.models.evidence import EvidenceView, ToolResult
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
    attribute_values: list[str] = Field(default_factory=list)


def _tool_result(result: list[UserAttribute]) -> ToolResult:
    evidence: list[EvidenceView] = []
    for attribute in result:
        metadata = UserAttributeMetadata(
            group_key=attribute.group_key,
            attribute_values=list(attribute.value),
        )
        evidence_view = EvidenceView(
            item_id=str(attribute.id),
            tool_name=TOOL_NAME_GET_USER_ATTRIBUTES,
            title=attribute.attribute_type,
            summary="Stored user attribute.",
            published_at=attribute.updated_at,
            source=TOOL_NAME_GET_USER_ATTRIBUTES,
            entity_type=TOOL_RESULT_TYPE_USER_ATTRIBUTE,
            llm_metadata=metadata.model_dump(exclude_none=True),
            raw_payload=attribute,
        )
        evidence.append(evidence_view)
    return ToolResult(result=result, evidence=evidence)


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
