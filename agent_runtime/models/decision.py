from __future__ import annotations

from enum import StrEnum
from typing import Any


class DecisionStatus(StrEnum):
    CONTINUE = "CONTINUE"
    SATISFIED = "SATISFIED"
    BLOCKED = "BLOCKED"


def normalize_decision_status(value: Any) -> DecisionStatus:
    normalized = str(value or "").strip().upper()
    if normalized in {"DONE", "SATISFIED", "COMPLETE", "COMPLETED"}:
        return DecisionStatus.SATISFIED
    if normalized == DecisionStatus.BLOCKED:
        return DecisionStatus.BLOCKED
    return DecisionStatus.CONTINUE
