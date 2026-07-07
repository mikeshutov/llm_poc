from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from llm.clients.embeddings import embed_text
from memories.models.memory_models import MemorySearchResult
from memories.models.memory_types import MEMORY_TYPE_DESCRIPTION, MemoryType
from memories.repository.repo_factory import get_memory_repo


class SearchUserMemoriesArgs(BaseModel):
    query: str = Field(..., description="Natural-language description of the user memory to find.")
    limit: int = Field(default=5, ge=1, le=10, description="Maximum number of matching memories to return.")
    is_active: bool | None = Field(default=True, description="Optional active-memory filter.")
    memory_type: MemoryType | None = Field(default=None, description=f"Optional memory type filter: {MEMORY_TYPE_DESCRIPTION}.")
    source: str | None = Field(default=None, description="Optional source filter.")


SEARCH_USER_MEMORIES_DESCRIPTION = f"""
Search persistent user memories by semantic similarity.

Required fields:
- query (string): Natural-language description of the memory to find.

Optional fields:
- limit (integer): Maximum number of matches to return. Defaults to 5.
- is_active (boolean): Optional active-memory filter.
- memory_type (string): {MEMORY_TYPE_DESCRIPTION}.
- source (string): Optional source filter.
"""


@tool(
    "search_user_memories",
    args_schema=SearchUserMemoriesArgs,
    description=SEARCH_USER_MEMORIES_DESCRIPTION,
)
def search_user_memories(
    query: str,
    limit: int = 5,
    is_active: bool | None = True,
    memory_type: MemoryType | None = None,
    source: str | None = None,
) -> list[MemorySearchResult]:
    query_embedding = embed_text(query)
    return get_memory_repo().search_memories(
        query_embedding=query_embedding,
        limit=limit,
        is_active=is_active,
        memory_type=memory_type,
        source=source,
    )
