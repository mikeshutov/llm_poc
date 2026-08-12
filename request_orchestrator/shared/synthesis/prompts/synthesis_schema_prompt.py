SYNTHESIS_SCHEMA = """
You MUST output valid JSON with this structure. Do not include raw markdown bullet points or dashes inside JSON string values as they break JSON parsing; use plain prose instead.

Ensure the `result` field is self-contained and final.
Ensure the `roundtrip_summary` field is a detailed plain-language diagnostic summary of the roundtrip outcome for history and retrieval.
The `roundtrip_summary` should usually be around 80 to 120 tokens and should be meaningfully richer than the top-level rolling conversation summary.
The `roundtrip_summary` should capture the user's request, the approach taken, the main evidence or tool findings used, important constraints or entities, and the final outcome.
Ensure the `tool_summary` object captures the tools actually used, what they produced, any key entities involved, and freshness if relevant.
Set `next_question` to one non-empty question or follow up statement.
Use `evidence_ids` only as structured attribution metadata. Do not mention evidence IDs, citation labels, or internal references inside `content`.
Every entry in `result` must be an object with exactly `content` and `evidence_ids`. Do not output `result` as an array of strings.

The `result` field MUST NOT contain:
- follow-up questions, invitations to continue or offers to provide more detail
- conversational closers (e.g., "let me know", "I can help", "if you'd like")

Expected Result Schema:

{
  "next_question": "<one follow up or clarifying question>",
  "result": [
    {
      "content": "<answer to the users question>",
      "evidence_ids": ["P1E1R1"]
    },
    {
      "content": "<additional paragraph if needed>",
      "evidence_ids": ["P1E2R1", "P1E3R1"]
    }
  ],
  "roundtrip_summary": "<detailed 80-120 token summary of what the user asked, what was done, what evidence or tools mattered, and what outcome was reached>",
  "tool_summary": {
    "used_tools": ["get_current_weather"],
    "produced": ["current_temperature", "wind_speed"],
    "entities": ["Toronto"],
    "freshness": "current as of 2026-06-29T09:00"
  }
}
"""
