EVALUATOR_SCHEMA = """{
  "satisfied": false,
  "relevant_evidence": ["E1", "E3"],
  "missing_information": [
    "Need current pricing for the top two products",
    "Need shipping availability in Canada"
  ],
  "refined_goal": "Find current Canadian pricing and availability for the two shortlisted products."
}

Rules:
- Set "satisfied" to true only when the gathered evidence is sufficient for synthesis to answer the goal well.
- Use "relevant_evidence" to list the step IDs whose evidence is actually relevant to the evaluation outcome.
- When "satisfied" is true, keep "missing_information" empty and "refined_goal" empty.
- When "satisfied" is false, list the concrete missing pieces and provide a refined_goal that the planner can act on next.
- Return JSON only.
"""
