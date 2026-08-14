from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, Field

EVALUATION_STATUS_SATISFIED: Final[Literal["SATISFIED"]] = "SATISFIED"
EVALUATION_STATUS_RETRYABLE: Final[Literal["RETRYABLE"]] = "RETRYABLE"
EVALUATION_STATUS_TERMINAL: Final[Literal["TERMINAL"]] = "TERMINAL"
TERMINAL_EVALUATION_STATUSES: Final[frozenset[str]] = frozenset({
    EVALUATION_STATUS_SATISFIED,
    EVALUATION_STATUS_TERMINAL,
})

EvaluationStatus = Literal[
    "SATISFIED",
    "RETRYABLE",
    "TERMINAL",
]


class EvaluationResult(BaseModel):
    status: EvaluationStatus = EVALUATION_STATUS_RETRYABLE
    relevant_evidence: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    refined_goal: str = ""
