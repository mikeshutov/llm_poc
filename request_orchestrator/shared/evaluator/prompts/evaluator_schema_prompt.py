from request_orchestrator.models.evaluation_result import (
    EVALUATION_STATUS_RETRYABLE,
    EVALUATION_STATUS_SATISFIED,
    EVALUATION_STATUS_TERMINAL,
)

EVALUATOR_SCHEMA = f"""Return a single JSON object with this shape:
{{
  "status": "{EVALUATION_STATUS_RETRYABLE}",
  "relevant_evidence": ["2f70c491-bcd8-5e2e-a520-1e0d3e8768c2"],
  "missing_information": [
    "the missing information to reach the goal"
  ],
  "refined_goal": "A refined goal to tackle the unresolved portion of the problem."
}}

Status values for the "status" field:
- {EVALUATION_STATUS_SATISFIED}
- {EVALUATION_STATUS_RETRYABLE}
- {EVALUATION_STATUS_TERMINAL}

Rules:
- Use "relevant_evidence" to list the evidence IDs whose evidence actually supports the evaluation outcome.
- {EVALUATION_STATUS_SATISFIED}: evidence is enough to answer well. Leave "missing_information" and "refined_goal" empty.
- {EVALUATION_STATUS_RETRYABLE}: a necessary gap prevents satisfying the stated goal. Identify only the specific information that is necessary and missing. Do not return RETRYABLE merely to improve completeness, confidence, variety, precision, or supporting detail when the existing evidence is already sufficient.
- {EVALUATION_STATUS_TERMINAL}: a necessary gap exists, but continued work is unlikely to resolve it. Keep "refined_goal" empty. Use "missing_information" only if it helps explain the limitation or failure.
- When referring to an entity in refined_goal include its ID.
"""
