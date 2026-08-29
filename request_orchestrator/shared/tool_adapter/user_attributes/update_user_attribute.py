from __future__ import annotations

from uuid import UUID

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from common.data import normalize_string_list
from llm.clients.embeddings import embed_text
from personalization.user_attributes.models.user_attribute_models import UserAttribute
from personalization.user_attributes.models.user_attribute_types import ATTRIBUTE_TYPE_COMPACT_DESCRIPTION, UserAttributeType
from personalization.user_attributes.repository.repo_factory import get_user_attribute_repo
from request_orchestrator.models.evidence import EvidenceView, ToolResult
from request_orchestrator.shared.runtime_context import get_current_user_id
from request_orchestrator.shared.tool_adapter.user_attribute_evidence_metadata import (
    USER_ATTRIBUTE_OPERATION_UPDATED,
    UserAttributeEvidenceMetadata,
)
from tool.constants import TOOL_NAME_UPDATE_USER_ATTRIBUTE
from tool.constants import TOOL_RESULT_TYPE_USER_ATTRIBUTE


class UpdateUserAttributeArgs(BaseModel):
    attribute_id: str = Field(..., description="The id of the attribute to update.")
    value: list[str] | None = Field(default=None, description="Updated concrete attribute values as a JSON array of strings.")
    attribute_type: UserAttributeType = Field(..., description=ATTRIBUTE_TYPE_COMPACT_DESCRIPTION)
    group_key: str | None = Field(default=None, description="Optional semantic grouping key for meaningful splits.")
    source: str | None = Field(default=None, description="Updated source value such as explicit, derived, or computed.")
    is_active: bool | None = Field(default=None, description="Whether the attribute should remain active.")
    confidence: float | None = Field(default=None, ge=0, le=1, description="Optional confidence score between 0 and 1.")
    importance: float | None = Field(default=None, ge=0, le=1, description="Optional importance score between 0 and 1.")


UPDATE_USER_ATTRIBUTE_DESCRIPTION = "Update an existing persistent user attribute."


def _value_text(value: list[str]) -> str:
    return "; ".join(normalize_string_list(value))


def _tool_result(result: UserAttribute) -> ToolResult:
    metadata = UserAttributeEvidenceMetadata(
        operation=USER_ATTRIBUTE_OPERATION_UPDATED,
        group_key=result.group_key,
        attribute_values=list(result.value),
    )
    evidence_view = EvidenceView(
        item_id=str(result.id),
        tool_name=TOOL_NAME_UPDATE_USER_ATTRIBUTE,
        title=result.attribute_type,
        summary="Stored user attribute.",
        published_at=result.updated_at,
        source=TOOL_NAME_UPDATE_USER_ATTRIBUTE,
        entity_type=TOOL_RESULT_TYPE_USER_ATTRIBUTE,
        llm_metadata=metadata.model_dump(exclude_none=True),
        raw_payload=result,
    )
    return ToolResult(
        result=result,
        evidence=[evidence_view],
    )




@tool(
    TOOL_NAME_UPDATE_USER_ATTRIBUTE,
    args_schema=UpdateUserAttributeArgs,
    description=UPDATE_USER_ATTRIBUTE_DESCRIPTION,
)
def update_user_attribute(
    attribute_id: str,
    attribute_type: UserAttributeType,
    value: list[str] | None = None,
    group_key: str | None = None,
    source: str | None = None,
    is_active: bool | None = None,
    confidence: float | None = None,
    importance: float | None = None,
) -> ToolResult:
    try:
        parsed_attribute_id = UUID(attribute_id)
    except ValueError:
        return ToolResult.error(f"Invalid attribute_id '{attribute_id}'.")

    attribute_embedding = embed_text(_value_text(value)) if value else None
    updated_attribute = get_user_attribute_repo().update_attribute(
        attribute_id=parsed_attribute_id,
        user_id=get_current_user_id(),
        value=value,
        attribute_embedding=attribute_embedding,
        attribute_type=attribute_type,
        group_key=group_key,
        source=source,
        is_active=is_active,
        confidence=confidence,
        importance=importance,
    )
    if updated_attribute is None:
        return ToolResult.error(f"No user attribute found with id '{attribute_id}'.")
    return _tool_result(updated_attribute)
