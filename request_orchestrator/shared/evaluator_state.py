from __future__ import annotations

from dataclasses import dataclass, field

from request_orchestrator.models.evaluation_result import (
    EvaluationStatus,
    EVALUATION_STATUS_RETRYABLE,
)


@dataclass
class EvaluatorNodeState:
    node_name: str = "evaluator"
    evaluation_status: EvaluationStatus = EVALUATION_STATUS_RETRYABLE
    missing_information: list[str] = field(default_factory=list)
