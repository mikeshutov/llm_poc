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
  "final_answer": "<sentence if one is available, otherwise null>",
  "needs_replan": false
}

Schema Rules:
- Set "needs_replan": true ONLY if previous tool results were insufficient and different tool calls are needed.
- Set "needs_replan": false when you already have results to to answer. This is also the default.
- Never repeat a tool call with identical arguments if it has already been called and provided as data.
"""

