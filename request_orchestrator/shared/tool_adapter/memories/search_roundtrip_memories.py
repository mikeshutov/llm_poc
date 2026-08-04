from __future__ import annotations

from uuid import UUID

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from conversation.models.conversation_models import RoundtripMemory
from conversation.repository.repo_factory import get_conversation_repo
from llm.clients.embeddings import embed_text


class SearchRoundtripMemoriesArgs(BaseModel):
    query: str = Field(..., description="Natural-language query describing the topic or specific mention to find in prior roundtrips.")
    conversation_ids: list[str] = Field(..., description="Conversation IDs returned from search_memories that should scope the roundtrip search.")
    limit: int = Field(default=5, ge=1, le=10, description="Maximum number of matching roundtrips to return.")


@tool(
    "search_roundtrip_memories",
    args_schema=SearchRoundtripMemoriesArgs,
    description="""
Search prior roundtrip summaries by semantic similarity within specific conversations.

Use search_memories first to identify relevant conversation_ids, then use this tool to find the specific historical exchanges most related to the topic.

Required fields:
- query (string): Natural-language description of the specific topic, mention, or exchange to find.
- conversation_ids (array of strings): One or more conversation IDs returned from search_memories.
- limit (integer, optional): Maximum number of matching roundtrips to return. Defaults to 5.
""",
)
def search_roundtrip_memories(query: str, conversation_ids: list[str], limit: int = 5) -> list[RoundtripMemory]:
    parsed_ids: list[UUID] = []
    for conversation_id in conversation_ids:
        try:
            parsed_ids.append(UUID(conversation_id))
        except ValueError:
            continue

    if not parsed_ids:
        return []

    query_embedding = embed_text(query)
    return get_conversation_repo().search_roundtrip_memories(
        query_embedding=query_embedding,
        conversation_ids=parsed_ids,
        limit=limit,
    )
