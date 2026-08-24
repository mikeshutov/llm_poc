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
    "Need current pricing for the top two products",
    "Need shipping availability in Canada"
  ],
  "refined_goal": "Find current Canadian pricing and availability for the two shortlisted products."
}}

Status values for the "status" field:
- {EVALUATION_STATUS_SATISFIED}
- {EVALUATION_STATUS_RETRYABLE}
- {EVALUATION_STATUS_TERMINAL}

Rules:
- Use "relevant_evidence" to list the evidence IDs whose evidence actually supports the evaluation outcome.
- {EVALUATION_STATUS_SATISFIED}: evidence is enough to answer well. Leave "missing_information" and "refined_goal" empty.
- {EVALUATION_STATUS_RETRYABLE}: evidence is not enough and there is still a meaningful next action. List the missing pieces and provide a refined_goal the planner can act on next.
- {EVALUATION_STATUS_TERMINAL}: no materially different useful action remains, or the evidence is enough to conclude the search failed or no match was found. Keep "refined_goal" empty. Use "missing_information" only if it helps explain the limitation or failure.
- When referring to an entity in refined_goal include its ID.
"""
