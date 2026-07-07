from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from memories.models.memory_models import Memory
from memories.models.memory_types import MEMORY_TYPE_DESCRIPTION, MemoryType
from memories.repository.repo_factory import get_memory_repo


class GetUserMemoriesArgs(BaseModel):
    limit: int = Field(default=10, ge=1, le=50, description="Maximum number of memories to return.")
    order_by: str = Field(default="updated_at", description="Sort field: created_at, updated_at, confidence, or importance.")
    descending: bool = Field(default=True, description="Whether to sort in descending order.")
    is_active: bool | None = Field(default=True, description="Optional active-memory filter.")
    memory_type: MemoryType | None = Field(default=None, description=f"Optional memory type filter: {MEMORY_TYPE_DESCRIPTION}.")
    source: str | None = Field(default=None, description="Optional source filter.")


GET_USER_MEMORIES_DESCRIPTION = f"""
List stored user memories with ordering and optional filters.

Optional fields:
- limit (integer): Maximum number of memories to return.
- order_by (string): created_at, updated_at, confidence, or importance.
- descending (boolean): Sort descending when true.
- is_active (boolean): Optional active-memory filter.
- memory_type (string): {MEMORY_TYPE_DESCRIPTION}.
- source (string): Optional source filter.
"""


@tool(
    "get_user_memories",
    args_schema=GetUserMemoriesArgs,
    description=GET_USER_MEMORIES_DESCRIPTION,
)
def get_user_memories(
    limit: int = 10,
    order_by: str = "updated_at",
    descending: bool = True,
    is_active: bool | None = True,
    memory_type: MemoryType | None = None,
    source: str | None = None,
) -> list[Memory]:
    return get_memory_repo().list_memories(
        limit=limit,
        order_by=order_by,
        descending=descending,
        is_active=is_active,
        memory_type=memory_type,
        source=source,
    )
