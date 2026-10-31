from __future__ import annotations

from enum import StrEnum


class DecisionStatus(StrEnum):
    CONTINUE = "CONTINUE"
    SATISFIED = "SATISFIED"
    BLOCKED = "BLOCKED"
