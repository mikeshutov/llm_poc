from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from conversation.models.conversation_models import ConversationMemory
from conversation.repository.repo_factory import get_conversation_repo
from llm.clients.embeddings import embed_text
from request_orchestrator.models.evidence import EvidenceView, HydratedEvidence, ToolResult
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
Search prior conversation memories by semantic similarity over conversation summaries.

Required fields:
- query (string): Natural language description of the memory or prior conversation you want to find.

Returns up to {DEFAULT_MEMORY_RESULT_LIMIT} relevant conversation memories for the current user.
Each result includes conversation_id, summary, last_used_date, and relevance_score.
""",
)
def search_memories(query: str) -> ToolResult:
    query_embedding = embed_text(query)
    memories = get_conversation_repo().search_conversation_memories(
        query_embedding=query_embedding,
        limit=DEFAULT_MEMORY_RESULT_LIMIT,
        user_id=get_current_user_id(),
    )
    hydrated_evidence: list[HydratedEvidence] = []
    evidence_views: list[EvidenceView] = []
    for memory in memories:
        metadata = MemorySearchMetadata(
            conversation_id=str(memory.conversation_id),
            relevance_score=memory.relevance_score,
        )
        hydrated = HydratedEvidence(
            item_id=str(memory.conversation_id),
            tool_name=TOOL_NAME_SEARCH_MEMORIES,
            title="Conversation Memory",
            summary=memory.summary,
            published_at=memory.last_used_date,
            source=TOOL_NAME_SEARCH_MEMORIES,
            entity_type=TOOL_RESULT_TYPE_MEMORY_RESULTS,
            metadata=metadata.model_dump(exclude_none=True),
            raw_payload=memory,
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
    return ToolResult(result=memories, evidence_views=evidence_views, hydrated_evidence=hydrated_evidence)
