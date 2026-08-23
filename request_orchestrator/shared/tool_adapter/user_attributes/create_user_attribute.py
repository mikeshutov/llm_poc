from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from common.data import normalize_string_list
from llm.clients.embeddings import embed_text
from personalization.user_attributes.models.user_attribute_models import UserAttribute
from personalization.user_attributes.models.user_attribute_types import ATTRIBUTE_TYPE_COMPACT_DESCRIPTION, UserAttributeType
from personalization.user_attributes.repository.repo_factory import get_user_attribute_repo
from request_orchestrator.models.evidence import EvidenceView, ToolResult
from request_orchestrator.shared.runtime_context import get_current_user_id
from tool.constants import TOOL_NAME_CREATE_USER_ATTRIBUTE
from tool.constants import TOOL_RESULT_TYPE_USER_ATTRIBUTE


class CreateUserAttributeArgs(BaseModel):
    value: list[str] = Field(..., description="Concrete attribute values as a JSON array of strings.")
    attribute_type: UserAttributeType = Field(..., description=ATTRIBUTE_TYPE_COMPACT_DESCRIPTION)
    group_key: str | None = Field(default=None, description="Optional semantic grouping key for meaningful splits.")
    source: str = Field(default="explicit", description="Attribute source such as explicit, derived, or computed.")
    confidence: float | None = Field(default=None, ge=0, le=1, description="Optional confidence score between 0 and 1.")
    importance: float | None = Field(default=None, ge=0, le=1, description="Optional importance score between 0 and 1.")


CREATE_USER_ATTRIBUTE_DESCRIPTION = "Create a persistent user attribute."


class UserAttributeMetadata(BaseModel):
    group_key: str | None = None
    source: str | None = None
    is_active: bool
    confidence: float | None = None
    importance: float | None = None


def _value_text(value: list[str]) -> str:
    return "; ".join(normalize_string_list(value))


def _tool_result(result: UserAttribute) -> ToolResult:
    summary = _value_text(result.value).strip() or "Stored user attribute."
    metadata = UserAttributeMetadata(
        group_key=result.group_key,
        source=result.source,
        is_active=result.is_active,
        confidence=result.confidence,
        importance=result.importance,
    )
    hydrated = EvidenceView(
        item_id=str(result.id),
        tool_name=TOOL_NAME_CREATE_USER_ATTRIBUTE,
        title=result.attribute_type,
        summary=summary,
        published_at=result.updated_at,
        source=TOOL_NAME_CREATE_USER_ATTRIBUTE,
        entity_type=TOOL_RESULT_TYPE_USER_ATTRIBUTE,
        llm_metadata=metadata.model_dump(exclude_none=True),
        raw_payload=result,
    )
    return ToolResult(
        result=result,
        evidence=[hydrated],
    )


@tool(
    TOOL_NAME_CREATE_USER_ATTRIBUTE,
    args_schema=CreateUserAttributeArgs,
    description=CREATE_USER_ATTRIBUTE_DESCRIPTION,
)
def create_user_attribute(
    value: list[str],
    attribute_type: UserAttributeType,
    group_key: str | None = None,
    source: str = "explicit",
    confidence: float | None = None,
    importance: float | None = None,
) -> ToolResult:
    return _tool_result(
        get_user_attribute_repo().create_attribute(
            value=value,
            user_id=get_current_user_id(),
            attribute_embedding=embed_text(_value_text(value)),
            attribute_type=attribute_type,
            group_key=group_key,
            source=source,
            confidence=confidence,
            importance=importance,
        )
    )
