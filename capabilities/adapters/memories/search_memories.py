from __future__ import annotations

from uuid import UUID

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from conversation.models.conversation_models import ConversationMemory, RoundtripMemory
from conversation.repository.repo_factory import get_conversation_repo
from llm.clients.embeddings import embed_text
from retrieval.evidence import EvidenceView, ToolResult
from agent_runtime.context import current_agent_state
from capabilities.adapters.memories.constants import DEFAULT_MEMORY_RESULT_LIMIT
from capabilities.registry.constants import TOOL_NAME_SEARCH_MEMORIES
from capabilities.registry.constants import TOOL_RESULT_TYPE_MEMORY_RESULTS


class SearchMemoriesArgs(BaseModel):
    query: str = Field(..., description="Natural-language query describing the memory to search for.")
    conversation_ids: list[str] | None = Field(
        default=None,
        description=(
            "Optional conversation IDs from a previous memory.search result. "
            "When provided, search detailed roundtrip messages within those conversations."
        ),
    )
    limit: int = Field(
        default=DEFAULT_MEMORY_RESULT_LIMIT,
        ge=1,
        le=10,
        description=f"Maximum number of memories to return. Defaults to {DEFAULT_MEMORY_RESULT_LIMIT}.",
    )


class MemorySearchMetadata(BaseModel):
    conversation_id: str
    relevance_score: float | None = None


class RoundtripMemoryMetadata(BaseModel):
    conversation_id: str
    roundtrip_id: str
    message_index: int
    created_at: str | None = None
    relevance_score: float | None = None


@tool(
    TOOL_NAME_SEARCH_MEMORIES,
    args_schema=SearchMemoriesArgs,
    description=f"""
Search prior conversation memories by semantic similarity. Omit conversation_ids to discover relevant conversations; provide conversation IDs from that result to search detailed roundtrip messages. Returns up to the requested limit of conversation or roundtrip memories.
""",
)
def search_memories(
    query: str,
    conversation_ids: list[str] | None = None,
    limit: int = DEFAULT_MEMORY_RESULT_LIMIT,
) -> ToolResult:
    query_embedding = embed_text(query)
    repo = get_conversation_repo()

    active_state = current_agent_state.get()
    active_user_id = active_state.execution_context.user_profile.user_id if active_state is not None else None
    if conversation_ids is not None:
        parsed_ids: list[UUID] = []
        for conversation_id in conversation_ids:
            try:
                parsed_ids.append(UUID(conversation_id))
            except ValueError:
                continue

        if not parsed_ids:
            return ToolResult(result=[], evidence=[])

        memories = repo.search_message_memories(
            query_embedding=query_embedding,
            conversation_ids=parsed_ids,
            limit=limit,
            user_id=active_user_id,
        )
        evidence: list[EvidenceView] = []
        for memory in memories:
            metadata = RoundtripMemoryMetadata(
                conversation_id=str(memory.conversation_id),
                roundtrip_id=str(memory.roundtrip_id),
                message_index=memory.message_index,
                created_at=memory.created_at,
                relevance_score=memory.relevance_score,
            )
            evidence.append(
                EvidenceView(
                    summary=(memory.roundtrip_summary or "Retrieved prior conversation memory.").strip(),
                    source=TOOL_NAME_SEARCH_MEMORIES,
        type=TOOL_RESULT_TYPE_MEMORY_RESULTS,
        metadata=metadata.model_dump(exclude_none=True),
        data={
            "item_id": str(memory.roundtrip_id),
            "title": f"Memory from message {memory.message_index}",
            "published_at": memory.created_at,
        })
            )
        return ToolResult(result=memories, evidence=evidence)

    roundtrip_matches = repo.search_roundtrip_memories(
        query_embedding=query_embedding,
        limit=limit,
        user_id=active_user_id,
    )
    best_by_conversation: dict[str, ConversationMemory] = {}
    for match in roundtrip_matches:
        conversation_id = str(match.conversation_id)
        conversation = repo.get_conversation(match.conversation_id)
        candidate = ConversationMemory(
            conversation_id=match.conversation_id,
            summary=(conversation.summary if conversation else "") or match.roundtrip_summary,
            last_used_date=str(conversation.created_at if conversation else match.created_at),
            relevance_score=match.relevance_score,
            matched_roundtrip_id=match.roundtrip_id,
            matched_message_id=match.matched_message_id,
        )
        previous = best_by_conversation.get(conversation_id)
        if previous is None or candidate.relevance_score < previous.relevance_score:
            best_by_conversation[conversation_id] = candidate
    memories = list(best_by_conversation.values())[:limit]
    evidence: list[EvidenceView] = []
    for memory in memories:
        metadata = MemorySearchMetadata(
            conversation_id=str(memory.conversation_id),
            relevance_score=memory.relevance_score,
        )
        evidence_view = EvidenceView(
            summary=memory.summary,
            source=TOOL_NAME_SEARCH_MEMORIES,
        type=TOOL_RESULT_TYPE_MEMORY_RESULTS,
        metadata=metadata.model_dump(exclude_none=True),
        data={
            "item_id": str(memory.conversation_id),
            "title": "Conversation Memory",
            "published_at": memory.last_used_date,
        })
        evidence.append(evidence_view)
    return ToolResult(result=memories, evidence=evidence)




