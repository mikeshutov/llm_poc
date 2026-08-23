from __future__ import annotations

from uuid import UUID

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from conversation.models.conversation_models import RoundtripMemory
from conversation.repository.repo_factory import get_conversation_repo
from llm.clients.embeddings import embed_text
from request_orchestrator.models.evidence import EvidenceView, ToolResult
from request_orchestrator.shared.runtime_context import get_current_user_id
from request_orchestrator.shared.tool_adapter.memories.constants import DEFAULT_MEMORY_RESULT_LIMIT
from tool.constants import TOOL_NAME_SEARCH_ROUNDTRIP_MEMORIES
from tool.constants import TOOL_RESULT_TYPE_MEMORY_RESULTS


class SearchRoundtripMemoriesArgs(BaseModel):
    query: str = Field(..., description="Natural-language query describing the topic or specific mention to find in prior roundtrips.")
    conversation_ids: list[str] = Field(..., description="Conversation IDs returned from search_memories that should scope the roundtrip search.")
    limit: int = Field(default=DEFAULT_MEMORY_RESULT_LIMIT, ge=1, le=10, description=f"Maximum number of matching roundtrips to return. Defaults to {DEFAULT_MEMORY_RESULT_LIMIT}.")


class RoundtripMemoryMetadata(BaseModel):
    conversation_id: str
    roundtrip_id: str
    message_index: int
    user_prompt: str | None = None
    created_at: str | None = None
    relevance_score: float | None = None


@tool(
    TOOL_NAME_SEARCH_ROUNDTRIP_MEMORIES,
    args_schema=SearchRoundtripMemoriesArgs,
    description=f"""
Search prior roundtrip summaries by semantic similarity within specific conversations.

Use search_memories first to identify relevant conversation_ids, then use this tool to find the specific historical exchanges most related to the topic.

Required fields:
- query (string): Natural-language description of the specific topic, mention, or exchange to find.
- conversation_ids (array of strings): One or more conversation IDs returned from search_memories.
- limit (integer, optional): Maximum number of matching roundtrips to return. Defaults to {DEFAULT_MEMORY_RESULT_LIMIT}.
""",
)
def search_roundtrip_memories(query: str, conversation_ids: list[str], limit: int = DEFAULT_MEMORY_RESULT_LIMIT) -> ToolResult:
    parsed_ids: list[UUID] = []
    for conversation_id in conversation_ids:
        try:
            parsed_ids.append(UUID(conversation_id))
        except ValueError:
            continue

    if not parsed_ids:
        return ToolResult(result=[], evidence=[])

    query_embedding = embed_text(query)
    memories = get_conversation_repo().search_roundtrip_memories(
        query_embedding=query_embedding,
        conversation_ids=parsed_ids,
        limit=limit,
        user_id=get_current_user_id(),
    )
    evidence: list[EvidenceView] = []
    for memory in memories:
        metadata = RoundtripMemoryMetadata(
            conversation_id=str(memory.conversation_id),
            roundtrip_id=str(memory.roundtrip_id),
            message_index=memory.message_index,
            user_prompt=memory.user_prompt,
            created_at=memory.created_at,
            relevance_score=memory.relevance_score,
        )
        hydrated = EvidenceView(
            item_id=str(memory.roundtrip_id),
            tool_name=TOOL_NAME_SEARCH_ROUNDTRIP_MEMORIES,
            title=f"Memory from message {memory.message_index}",
            summary=(memory.roundtrip_summary or memory.generated_response or "").strip(),
            published_at=memory.created_at,
            source=TOOL_NAME_SEARCH_ROUNDTRIP_MEMORIES,
            entity_type=TOOL_RESULT_TYPE_MEMORY_RESULTS,
            llm_metadata=metadata.model_dump(exclude_none=True),
            raw_payload=memory,
        )
        evidence.append(hydrated)
    return ToolResult(result=memories, evidence=evidence)
