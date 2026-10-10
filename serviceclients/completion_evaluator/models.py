from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CompletionEvaluatorRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    premise: str
    hypothesis: str


class CompletionEvaluatorResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contradiction: float = Field(ge=0, le=1)
    neutral: float = Field(ge=0, le=1)
    entailment: float = Field(ge=0, le=1)
