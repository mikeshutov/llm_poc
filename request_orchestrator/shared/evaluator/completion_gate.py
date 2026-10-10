from __future__ import annotations

import json
import logging
import os

from serviceclients.completion_evaluator import CompletionEvaluatorClient, CompletionEvaluatorResult
from request_orchestrator.models.agent_prompt import EvidenceStep
from request_orchestrator.models.agent_state import AgentState

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

COMPLETION_EVALUATOR_ENTAILMENT_THRESHOLD = float(
    os.getenv("COMPLETION_EVALUATOR_ENTAILMENT_THRESHOLD", "0.85")
)
COMPLETION_EVALUATOR_MAX_EVIDENCE_CHARS = max(
    512, int(os.getenv("COMPLETION_EVALUATOR_MAX_EVIDENCE_CHARS", "12000"))
)


def _build_premise(evidence_steps: list[EvidenceStep]) -> str:
    records = [
        {
            "type": step.type,
            "no_results": step.no_results,
            "evidence": [evidence.to_evaluator_view() for evidence in step.evidence],
        }
        for step in evidence_steps
    ]
    return json.dumps(records, ensure_ascii=True, default=str)[:COMPLETION_EVALUATOR_MAX_EVIDENCE_CHARS]


def try_completion_evaluation(
    state: AgentState,
    evidence_steps: list[EvidenceStep],
    *,
    client: CompletionEvaluatorClient | None = None,
) -> bool:
    """Return whether the evidence clears the completion threshold."""
    if not evidence_steps or not any(step.evidence for step in evidence_steps):
        return False

    resolved_client = client or CompletionEvaluatorClient()
    premise = _build_premise(evidence_steps)
    hypothesis = (
        "The provided evidence is sufficient to answer the user's request: "
        + state.inputs.task.strip()
    )
    logger.warning(
        "Completion evaluation payload: premise=%s hypothesis=%s",
        premise,
        hypothesis,
    )
    result: CompletionEvaluatorResult = resolved_client.evaluate(
        premise,
        hypothesis,
    )
    satisfied = result.entailment >= COMPLETION_EVALUATOR_ENTAILMENT_THRESHOLD
    logger.warning(
        "Completion evaluation scores: contradiction=%.4f neutral=%.4f entailment=%.4f "
        "threshold=%.4f fast_path=%s",
        result.contradiction,
        result.neutral,
        result.entailment,
        COMPLETION_EVALUATOR_ENTAILMENT_THRESHOLD,
        satisfied,
    )
    return satisfied
