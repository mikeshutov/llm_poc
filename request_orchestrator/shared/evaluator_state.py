from __future__ import annotations

from dataclasses import dataclass

from request_orchestrator.models.evaluation_result import (
    EvaluationStatus,
    EVALUATION_STATUS_RETRYABLE,
)


@dataclass
class EvaluatorNodeState:
    node_name: str = "evaluator"
    evaluation_status: EvaluationStatus = EVALUATION_STATUS_RETRYABLE
    goal_reached: bool = False
