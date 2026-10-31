from __future__ import annotations

from dataclasses import dataclass, field

from agent_runtime.constants import NodeName
from agent_runtime.models.evaluation_result import (
    EvaluationStatus,
    EVALUATION_STATUS_RETRYABLE,
)


@dataclass
class EvaluatorNodeState:
    node_name: str = NodeName.EVALUATOR.value
    evaluation_status: EvaluationStatus = EVALUATION_STATUS_RETRYABLE
    missing_information: list[str] = field(default_factory=list)
