from __future__ import annotations

from pydantic import BaseModel, Field


class RerankerResult(BaseModel):
    ranked_ids: list[str] = Field(default_factory=list)
