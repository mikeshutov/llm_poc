PLANNER_SCHEMA = """
You MUST output valid JSON with this structure:
{
  "steps": [
    {
      "id": "E1",
      "plan": "<one sentence>",
      "tool": "<TOOL_NAME>",
      "args": { <JSON object> }
    }
  ],
  "final_answer": null,
  "needs_replan": false
}

Schema Rules:
- Return tool steps only when there is a concrete useful tool call to make.
- If there is nothing actionable to call, return an empty "steps" list.
- Always set "final_answer" to null.
- Always set "needs_replan" to false.
- Return JSON only.
"""
