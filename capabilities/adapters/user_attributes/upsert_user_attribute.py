from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from common.data import normalize_string_list
from llm.clients.embeddings import embed_text
from personalization.user_attributes.models.user_attribute_models import UserAttribute
from personalization.user_attributes.models.user_attribute_types import ATTRIBUTE_TYPE_COMPACT_DESCRIPTION, UserAttributeType
from personalization.user_attributes.repository.repo_factory import get_user_attribute_repo
from retrieval.evidence import EvidenceView, ToolResult
from agent_runtime.context import get_current_user_id
from capabilities.adapters.user_attributes.user_attribute_evidence_metadata import (
    USER_ATTRIBUTE_OPERATION_UPSERTED,
    UserAttributeEvidenceMetadata,
)
from capabilities.registry.constants import TOOL_NAME_UPSERT_USER_ATTRIBUTE, TOOL_RESULT_TYPE_USER_ATTRIBUTE


class UpsertUserAttributeArgs(BaseModel):
    value: list[str] = Field(..., description="Concrete attribute values as a JSON array of strings.")
    attribute_type: UserAttributeType = Field(..., description=ATTRIBUTE_TYPE_COMPACT_DESCRIPTION)
    group_key: str | None = Field(default=None, description="Optional semantic grouping key for meaningful splits.")
    source: str = Field(default="explicit", description="Attribute source such as explicit, derived, or computed.")
    confidence: float | None = Field(default=None, ge=0, le=1, description="Optional confidence score between 0 and 1.")
    importance: float | None = Field(default=None, ge=0, le=1, description="Optional importance score between 0 and 1.")


@tool(
    TOOL_NAME_UPSERT_USER_ATTRIBUTE,
    args_schema=UpsertUserAttributeArgs,
    description=(
        "Create or update a persistent user attribute. Values are composed into the existing "
        "attribute bucket automatically; do not search for an existing attribute first."
    ),
)
def upsert_user_attribute(
    value: list[str],
    attribute_type: UserAttributeType,
    group_key: str | None = None,
    source: str = "explicit",
    confidence: float | None = None,
    importance: float | None = None,
) -> ToolResult:
    normalized_value = normalize_string_list(value)
    result: UserAttribute = get_user_attribute_repo().upsert_attribute(
        value=normalized_value,
        user_id=get_current_user_id(),
        attribute_embedding=embed_text("; ".join(normalized_value)),
        attribute_type=attribute_type,
        group_key=group_key,
        source=source,
        confidence=confidence,
        importance=importance,
    )
    metadata = UserAttributeEvidenceMetadata(
        operation=USER_ATTRIBUTE_OPERATION_UPSERTED,
        group_key=result.group_key,
        attribute_values=list(result.value),
    )
    evidence = EvidenceView(
        item_id=str(result.id),
        tool_name=TOOL_NAME_UPSERT_USER_ATTRIBUTE,
        title=result.attribute_type,
        summary="Stored and composed user attribute.",
        published_at=result.updated_at,
        source=TOOL_NAME_UPSERT_USER_ATTRIBUTE,
        entity_type=TOOL_RESULT_TYPE_USER_ATTRIBUTE,
        llm_metadata=metadata.model_dump(exclude_none=True),
        raw_payload=result,
    )
    return ToolResult(result=result, evidence=[evidence])
