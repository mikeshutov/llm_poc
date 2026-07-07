from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from llm.clients.embeddings import embed_text
from memories.models.memory_models import Memory
from memories.models.memory_types import MEMORY_TYPE_DESCRIPTION, MemoryType
from memories.repository.repo_factory import get_memory_repo


class CreateUserMemoryArgs(BaseModel):
    memory_text: str = Field(..., description="The user memory text to store.")
    memory_type: MemoryType = Field(..., description=f"Memory type: {MEMORY_TYPE_DESCRIPTION}.")
    source: str = Field(default="explicit", description="Memory source such as explicit, derived, or computed.")
    confidence: float | None = Field(default=None, ge=0, le=1, description="Optional confidence score between 0 and 1.")
    importance: float | None = Field(default=None, ge=0, le=1, description="Optional importance score between 0 and 1.")


CREATE_USER_MEMORY_DESCRIPTION = f"""
Create a persistent user memory for future recall.

Required fields:
- memory_text (string): The fact, preference, instruction, or stable detail to remember.
- memory_type (string): {MEMORY_TYPE_DESCRIPTION}.

Optional fields:
- source (string): explicit, derived, or computed. Defaults to explicit.
- confidence (number): Optional 0..1 confidence score.
- importance (number): Optional 0..1 importance score.
"""


@tool(
    "create_user_memory",
    args_schema=CreateUserMemoryArgs,
    description=CREATE_USER_MEMORY_DESCRIPTION,
)
def create_user_memory(
    memory_text: str,
    memory_type: MemoryType,
    source: str = "explicit",
    confidence: float | None = None,
    importance: float | None = None,
) -> Memory:
    return get_memory_repo().create_memory(
        memory_text=memory_text,
        memory_embedding=embed_text(memory_text),
        memory_type=memory_type,
        source=source,
        confidence=confidence,
        importance=importance,
    )
