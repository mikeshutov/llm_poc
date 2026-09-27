from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class RerankerCandidate(BaseModel):
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
    candidates: list[RerankerCandidate] = Field(default_factory=list)


class RerankerResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: list[RerankerScore] = Field(default_factory=list)
