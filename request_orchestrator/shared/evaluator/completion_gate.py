from __future__ import annotations

import logging
import os
import json

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
    records: list[str] = []
    for step in evidence_steps:
        for evidence in step.evidence:
            fields = evidence.to_candidate().fields
            name = evidence.title.strip() or evidence.item_id.strip() or str(evidence.id)
            available_fields = [
                key
                for key, value in fields.items()
                if key != "title" and value not in (None, "", [], {})
            ]
            field_parts = []
            evaluation_fields = set(step.evaluation_fields)
            for key in available_fields:
                value = fields.get(key)
                if key in evaluation_fields:
                    field_parts.append(f"{key}={json.dumps(value, ensure_ascii=True, default=str)}")
                else:
                    field_parts.append(key)
            records.append(f"{name} with {', '.join(field_parts)}" if field_parts else name)
    return "\n".join(records)[:COMPLETION_EVALUATOR_MAX_EVIDENCE_CHARS]


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
    hypotheses = [step.hypothesis.strip() for step in evidence_steps if step.hypothesis.strip()]
    hypothesis = "; ".join(dict.fromkeys(hypotheses)) or (
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
