from __future__ import annotations

import os
import json
from time import perf_counter

from common.logging import record_node_log
from request_orchestrator.constants import NodeName
from serviceclients.completion_evaluator import CompletionEvaluatorClient, CompletionEvaluatorResult
from request_orchestrator.models.agent_prompt import EvidenceStep
from request_orchestrator.models.agent_state import AgentState

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
    started_at = perf_counter()
    record_node_log(
        node_name=NodeName.COMPLETION_EVALUATOR,
        stage="evaluate",
        event_type="completion_evaluation",
        message="Completion evaluation started",
        roundtrip_id=state.execution_context.roundtrip_id,
        payload={
            "evidence_step_count": len(evidence_steps),
            "evidence_count": sum(len(step.evidence) for step in evidence_steps),
            "premise_preview": premise[:2_000],
            "premise_length": len(premise),
            "hypothesis_preview": hypothesis[:2_000],
            "hypothesis_length": len(hypothesis),
        },
    )
    try:
        result: CompletionEvaluatorResult = resolved_client.evaluate(
            premise,
            hypothesis,
        )
    except Exception as exc:
        record_node_log(
            node_name=NodeName.COMPLETION_EVALUATOR,
            stage="evaluate",
            event_type="completion_evaluation",
            level="error",
            message="Completion evaluation failed",
            roundtrip_id=state.execution_context.roundtrip_id,
            payload={"error": str(exc)},
            duration_ms=int((perf_counter() - started_at) * 1000),
        )
        raise
    satisfied = result.entailment >= COMPLETION_EVALUATOR_ENTAILMENT_THRESHOLD
    record_node_log(
        node_name=NodeName.COMPLETION_EVALUATOR,
        stage="evaluate",
        event_type="completion_evaluation",
        message="Completion evaluation completed",
        roundtrip_id=state.execution_context.roundtrip_id,
        payload={
            "contradiction": result.contradiction,
            "neutral": result.neutral,
            "entailment": result.entailment,
            "threshold": COMPLETION_EVALUATOR_ENTAILMENT_THRESHOLD,
            "satisfied": satisfied,
        },
        duration_ms=int((perf_counter() - started_at) * 1000),
    )
    return satisfied
