from __future__ import annotations

from typing import Literal, get_args

MemoryType = Literal["preference", "fact", "constraint", "instruction", "relationship", "other"]

MEMORY_TYPE_VALUES = tuple(get_args(MemoryType))
MEMORY_TYPE_DESCRIPTION = ", ".join(MEMORY_TYPE_VALUES)
