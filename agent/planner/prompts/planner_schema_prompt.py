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
- Set "needs_replan": true if the current plan alone is not sufficient to answer the task and further tool calls will be needed after these results come back.
- Set "needs_replan": false when the current plan steps are expected to produce enough data to answer the task when considering the full context. This is the default.
- Never repeat a tool call with identical arguments if it has already been called and provided as data.
"""

