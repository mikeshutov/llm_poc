from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from conversation.models.conversation_models import ConversationMemory
from conversation.repository.repo_factory import get_conversation_repo
from llm.clients.embeddings import embed_text
from request_orchestrator.models.evidence import EvidenceView, ToolResult
from request_orchestrator.shared.runtime_context import get_current_user_id
from request_orchestrator.shared.tool_adapter.memories.constants import DEFAULT_MEMORY_RESULT_LIMIT
from tool.constants import TOOL_NAME_SEARCH_MEMORIES
from tool.constants import TOOL_RESULT_TYPE_MEMORY_RESULTS


class SearchMemoriesArgs(BaseModel):
    query: str = Field(..., description="Natural-language query describing the memory to search for.")


class MemorySearchMetadata(BaseModel):
    conversation_id: str
    relevance_score: float | None = None


@tool(
    TOOL_NAME_SEARCH_MEMORIES,
    args_schema=SearchMemoriesArgs,
    description=f"""
Search prior conversation memories by semantic similarity over roundtrip summaries.
Results are grouped by conversation; use search_roundtrip_memories to retrieve the full matching exchanges.


Returns up to {DEFAULT_MEMORY_RESULT_LIMIT} relevant conversation memories for the current user.
""",
)
def search_memories(query: str) -> ToolResult:
    query_embedding = embed_text(query)
    repo = get_conversation_repo()
    roundtrip_matches = repo.search_roundtrip_memories(
        query_embedding=query_embedding,
        limit=DEFAULT_MEMORY_RESULT_LIMIT,
        user_id=get_current_user_id(),
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
    memories = list(best_by_conversation.values())[:DEFAULT_MEMORY_RESULT_LIMIT]
    evidence: list[EvidenceView] = []
    for memory in memories:
        metadata = MemorySearchMetadata(
            conversation_id=str(memory.conversation_id),
            relevance_score=memory.relevance_score,
        )
        evidence_view = EvidenceView(
            item_id=str(memory.conversation_id),
            tool_name=TOOL_NAME_SEARCH_MEMORIES,
            title="Conversation Memory",
            summary=memory.summary,
            published_at=memory.last_used_date,
            source=TOOL_NAME_SEARCH_MEMORIES,
            entity_type=TOOL_RESULT_TYPE_MEMORY_RESULTS,
            llm_metadata=metadata.model_dump(exclude_none=True),
            raw_payload=memory,
        )
        evidence.append(evidence_view)
    return ToolResult(result=memories, evidence=evidence)
