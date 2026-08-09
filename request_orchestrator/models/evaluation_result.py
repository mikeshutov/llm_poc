from __future__ import annotations

from pydantic import BaseModel, Field


class EvaluationResult(BaseModel):
    satisfied: bool = False
    relevant_evidence: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    refined_goal: str = ""
