from __future__ import annotations

from uuid import UUID

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from llm.clients.embeddings import embed_text
from memories.models.memory_models import Memory
from memories.models.memory_types import MEMORY_TYPE_DESCRIPTION, MemoryType
from memories.repository.repo_factory import get_memory_repo


class UpdateUserMemoryArgs(BaseModel):
    memory_id: str = Field(..., description="The id of the memory to update.")
    memory_text: str | None = Field(default=None, description="Updated memory text.")
    memory_type: MemoryType | None = Field(default=None, description=f"Updated memory type: {MEMORY_TYPE_DESCRIPTION}.")
    source: str | None = Field(default=None, description="Updated source value such as explicit, derived, or computed.")
    is_active: bool | None = Field(default=None, description="Whether the memory should remain active.")
    confidence: float | None = Field(default=None, ge=0, le=1, description="Optional confidence score between 0 and 1.")
    importance: float | None = Field(default=None, ge=0, le=1, description="Optional importance score between 0 and 1.")


UPDATE_USER_MEMORY_DESCRIPTION = f"""
Update an existing persistent user memory.

Required fields:
- memory_id (string): The memory id to update.

Optional fields:
- memory_text (string): Updated memory text.
- memory_type (string): {MEMORY_TYPE_DESCRIPTION}.
- source (string): Updated source value.
- is_active (boolean): Set false to deactivate a memory.
- confidence (number): Optional 0..1 confidence score.
- importance (number): Optional 0..1 importance score.
"""


@tool(
    "update_user_memory",
    args_schema=UpdateUserMemoryArgs,
    description=UPDATE_USER_MEMORY_DESCRIPTION,
)
def update_user_memory(
    memory_id: str,
    memory_text: str | None = None,
    memory_type: MemoryType | None = None,
    source: str | None = None,
    is_active: bool | None = None,
    confidence: float | None = None,
    importance: float | None = None,
) -> Memory | None:
    try:
        parsed_memory_id = UUID(memory_id)
    except ValueError:
        return None

    memory_embedding = embed_text(memory_text) if memory_text else None
    return get_memory_repo().update_memory(
        memory_id=parsed_memory_id,
        memory_text=memory_text,
        memory_embedding=memory_embedding,
        memory_type=memory_type,
        source=source,
        is_active=is_active,
        confidence=confidence,
        importance=importance,
    )
