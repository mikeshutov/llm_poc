from __future__ import annotations

from typing import Any
from uuid import UUID

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from llm.conversation_model_config import ConversationModelConfig
from conversation.repository.repo_factory import get_conversation_repo
from request_orchestrator.models.evidence import EvidenceView, ToolResult
from request_orchestrator.shared.runtime_context import get_current_user_id
from tool.constants import TOOL_NAME_GET_MEMORY_DETAIL
from tool.constants import TOOL_RESULT_TYPE_MEMORY_DETAIL


class GetMemoryDetailArgs(BaseModel):
    roundtrip_id: str = Field(..., description="The roundtrip_id returned from search_roundtrip_memories.")


class GetMemoryDetailResult(BaseModel):
    error: str = ""
    memory_type: str = ""
    title: str = ""
    summary: str = ""
    conversation_id: str = ""
    roundtrip_id: str = ""
    message_index: int = 0
    roundtrip_summary: str = ""
    created_at: str = ""
    model: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    messages: list[dict[str, Any]] = Field(default_factory=list)


class MemoryDetailEvidenceMetadata(BaseModel):
    resolved_model_config: ConversationModelConfig | None = None
    conversation_id: str = ""
    roundtrip_id: str = ""
    message_index: int = 0
    created_at: str = ""
    model: str = ""


@tool(
    TOOL_NAME_GET_MEMORY_DETAIL,
    args_schema=GetMemoryDetailArgs,
    description="""
Retrieve the full detail for a previously found roundtrip memory by roundtrip_id.
Use after search_roundtrip_memories when you need the exact prior exchange, including the original user prompt and generated response.


""",
)
def get_memory_detail(roundtrip_id: str) -> ToolResult:
    try:
        parsed_id = UUID(roundtrip_id)
    except ValueError:
        return ToolResult(
            result=GetMemoryDetailResult(error=f"Invalid roundtrip_id '{roundtrip_id}'."),

            evidence=[],
        )

    repo = get_conversation_repo()
    roundtrip = repo.get_roundtrip_for_user(
        parsed_id,
        user_id=get_current_user_id(),
    )
    if roundtrip is None:
        return ToolResult(
            result=GetMemoryDetailResult(error=f"No memory found with roundtrip_id '{roundtrip_id}'."),

            evidence=[],
        )

    roundtrip_summary = (roundtrip.roundtrip_summary or "").strip()
    messages = repo.list_messages(roundtrip.id, include_internal=True)
    user_message = next((message for message in messages if message.kind == "user"), None)
    synthesis = next((message for message in reversed(messages)
                      if message.metadata.get("stage") == "synthesis"), None)
    generated_response = synthesis.content.strip() if synthesis else ""
    result = GetMemoryDetailResult(
        memory_type="roundtrip",
        title=f"Memory detail for message {roundtrip.message_index}",
        summary=roundtrip_summary or generated_response or "Retrieved prior conversation memory detail.",
        conversation_id=str(roundtrip.conversation_id),
        roundtrip_id=str(roundtrip.id),
        message_index=roundtrip.message_index,
        roundtrip_summary=roundtrip.roundtrip_summary or "",
        created_at=str(roundtrip.created_at),
        model=str(synthesis.metadata.get("model", "")) if synthesis else "",
        metadata=roundtrip.metadata,
        messages=[
            {
                "id": str(message.id),
                "sequence": message.sequence,
                "kind": message.kind,
                "visibility": message.visibility,
                "role": message.role,
                "content": message.content,
                "payload": message.payload,
                "metadata": message.metadata,
                "created_at": str(message.created_at),
            }
            for message in repo.list_messages(roundtrip.id, include_internal=True)
        ],
    )
    metadata = MemoryDetailEvidenceMetadata(
        resolved_model_config=_resolved_model_config(result.metadata),
        conversation_id=result.conversation_id,
        roundtrip_id=result.roundtrip_id,
        message_index=result.message_index,
        created_at=result.created_at,
        model=result.model,
    )
    evidence_view = EvidenceView(
        item_id=result.roundtrip_id,
        tool_name=TOOL_NAME_GET_MEMORY_DETAIL,
        title=result.title.strip() or "Memory Detail",
        summary=result.summary.strip() or "Retrieved prior conversation memory detail.",
        published_at=result.created_at.strip(),
        source=TOOL_NAME_GET_MEMORY_DETAIL,
        entity_type=TOOL_RESULT_TYPE_MEMORY_DETAIL,
        llm_metadata=metadata.model_dump(mode="json", exclude_none=True),
        raw_payload=result,
    )
    return ToolResult(
        result=result,
        evidence=[evidence_view],
    )


def _resolved_model_config(metadata: dict[str, Any]) -> ConversationModelConfig | None:
    value = metadata.get("resolved_model_config")
    return ConversationModelConfig.model_validate(value) if isinstance(value, dict) else None
