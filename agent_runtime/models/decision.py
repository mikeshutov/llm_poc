from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class DecisionStatus(StrEnum):
    CONTINUE = "CONTINUE"
    SATISFIED = "SATISFIED"
    BLOCKED = "BLOCKED"


class DecisionResponse(BaseModel):
    """The structured response envelope returned by a decision subagent."""

    status: DecisionStatus
    message: str
