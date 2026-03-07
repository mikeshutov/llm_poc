SYNTHESIS_SCHEMA = """
You MUST output valid JSON with this structure:
The result needs to be an array of markdowns.

{
  "follow_up": "<one follow up question to drive engagement>",
  "result": ["<answer to the users question>","<if complex question additional paragraphs appear in array>"],
  "clarifying_question":"<a question to clarify the users desire for vague requests>"
}
"""
