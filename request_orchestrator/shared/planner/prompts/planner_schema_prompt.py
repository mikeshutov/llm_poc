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
  "status": "ready",
  "reason": ""
}

Schema Rules:
- Return tool steps only when there is a concrete useful tool call to make.
- If there is nothing actionable to call, return an empty "steps" list with status "ready".
- If the goal cannot proceed because a required capability is unavailable, return an empty "steps" list with status "blocked" and a short reason.
- Do not use outputs from one planned step as arguments for another planned step. Each step must be independently runnable from the provided args.
- You may use already-available tool results from previous work or previous iterations when choosing the next step.
- Return JSON only.
"""
