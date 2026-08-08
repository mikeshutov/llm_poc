from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from reranker.models.candidate_content import CandidateContent


class Candidate(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    id: str
    candidate_type: str | None = None
    title: str | None = None
    content: CandidateContent = Field(default_factory=CandidateContent)
    attributes: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float | None = None
    embedding: list[float] = Field(default_factory=list)
