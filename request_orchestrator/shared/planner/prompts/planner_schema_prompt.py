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
  ]
}

Schema Rules:
- Return tool steps only when there is a concrete useful tool call to make.
- If there is nothing actionable to call, return an empty "steps" list.
- Return JSON only.
"""
