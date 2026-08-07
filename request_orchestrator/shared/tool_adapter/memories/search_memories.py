from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from conversation.models.conversation_models import ConversationMemory
from conversation.repository.repo_factory import get_conversation_repo
from llm.clients.embeddings import embed_text
from request_orchestrator.shared.tool_adapter.memories.constants import DEFAULT_MEMORY_RESULT_LIMIT


class SearchMemoriesArgs(BaseModel):
    query: str = Field(..., description="Natural-language query describing the memory to search for.")


@tool(
    "search_memories",
    args_schema=SearchMemoriesArgs,
    description=f"""
Search prior conversation memories by semantic similarity over conversation summaries.

Required fields:
- query (string): Natural language description of the memory or prior conversation you want to find.

Returns up to {DEFAULT_MEMORY_RESULT_LIMIT} relevant conversation memories for the current user.
Each result includes conversation_id, summary, last_used_date, and relevance_score.
""",
)
def search_memories(query: str) -> list[ConversationMemory]:
    query_embedding = embed_text(query)
    return get_conversation_repo().search_conversation_memories(
        query_embedding=query_embedding,
        limit=DEFAULT_MEMORY_RESULT_LIMIT,
    )
