from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    fields: dict[str, Any] = Field(default_factory=dict)




class RerankerRequestCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    text: str


class RerankerScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    score: float


class RerankerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    candidates: list[RerankerRequestCandidate] = Field(default_factory=list)


class RerankerResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: list[RerankerScore] = Field(default_factory=list)
