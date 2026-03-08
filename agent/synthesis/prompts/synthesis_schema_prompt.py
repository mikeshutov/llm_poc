SYNTHESIS_SCHEMA = """
You MUST output valid JSON with this structure:
The result needs to be an array of markdowns.

Ensure the `result` field is self-contained and final.

The `result` field MUST NOT contain:
- follow-up questions
- invitations to continue
- offers to provide more detail
- conversational closers (e.g., "let me know", "I can help", "if you'd like")

{
  "follow_up": "<one follow up question to drive engagement>",
  "result": ["<answer to the users question>","<if complex question additional paragraphs appear in array no follow ups or conversation closures>"],
  "clarifying_question":"<a question to clarify the users desire for vague requests>"
}
"""
