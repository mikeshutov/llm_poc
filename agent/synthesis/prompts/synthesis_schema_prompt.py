SYNTHESIS_SCHEMA = """
You MUST output valid JSON with this structure. Do not include raw markdown bullet points or dashes inside JSON string values as they break JSON parsing — use plain prose instead.

Ensure the `result` field is self-contained and final.

The `result` field MUST NOT contain:
- follow-up questions
- invitations to continue
- offers to provide more detail
- conversational closers (e.g., "let me know", "I can help", "if you'd like")

{
  "follow_up": "<one follow up question to drive engagement>",
  "result": ["<answer to the users question>", "<additional paragraph if needed>"],
  "clarifying_question": "<a question to clarify the users desire for vague requests>"
}
"""
