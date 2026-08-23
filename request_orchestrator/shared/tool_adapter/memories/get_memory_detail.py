from __future__ import annotations

from typing import Any
from uuid import UUID

from langchain_core.tools import tool
from pydantic import BaseModel, Field

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
    user_prompt: str = ""
    generated_response: str = ""
    roundtrip_summary: str = ""
    created_at: str = ""
    model: str = ""
    response_payload: dict[str, Any] = Field(default_factory=dict)
    parsed_query: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


@tool(
    TOOL_NAME_GET_MEMORY_DETAIL,
    args_schema=GetMemoryDetailArgs,
    description="""
Retrieve the full detail for a previously found roundtrip memory by roundtrip_id.
Use after search_roundtrip_memories when you need the exact prior exchange, including the original user prompt and generated response.

Required fields:
- roundtrip_id (string): The roundtrip_id returned from search_roundtrip_memories.

Example valid calls:
{"roundtrip_id": "3f2a1b4c-..."}
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

    roundtrip = get_conversation_repo().get_roundtrip_for_user(
        parsed_id,
        user_id=get_current_user_id(),
    )
    if roundtrip is None:
        return ToolResult(
            result=GetMemoryDetailResult(error=f"No memory found with roundtrip_id '{roundtrip_id}'."),

            evidence=[],
        )

    roundtrip_summary = (roundtrip.roundtrip_summary or "").strip()
    generated_response = (roundtrip.generated_response or "").strip()
    result = GetMemoryDetailResult(
        memory_type="roundtrip",
        title=f"Memory detail for message {roundtrip.message_index}",
        summary=roundtrip_summary or generated_response or "Retrieved prior conversation memory detail.",
        conversation_id=str(roundtrip.conversation_id),
        roundtrip_id=str(roundtrip.id),
        message_index=roundtrip.message_index,
        user_prompt=roundtrip.user_prompt,
        generated_response=roundtrip.generated_response,
        roundtrip_summary=roundtrip.roundtrip_summary or "",
        created_at=str(roundtrip.created_at),
        model=roundtrip.model or "",
        response_payload=roundtrip.response_payload or {},
        parsed_query=roundtrip.parsed_query or {},
        metadata=roundtrip.metadata or {},
    )
    metadata = dict(result.metadata)
    metadata.update(
        {
            "conversation_id": result.conversation_id,
            "roundtrip_id": result.roundtrip_id,
            "message_index": result.message_index,
            "created_at": result.created_at,
            "model": result.model,
        }
    )
    evidence_view = EvidenceView(
        item_id=result.roundtrip_id,
        tool_name=TOOL_NAME_GET_MEMORY_DETAIL,
        title=result.title.strip() or "Memory Detail",
        summary=result.summary.strip() or "Retrieved prior conversation memory detail.",
        published_at=result.created_at.strip(),
        source=TOOL_NAME_GET_MEMORY_DETAIL,
        entity_type=TOOL_RESULT_TYPE_MEMORY_DETAIL,
        llm_metadata=metadata,
        raw_payload=result,
    )
    return ToolResult(
        result=result,
        evidence=[evidence_view],
    )
