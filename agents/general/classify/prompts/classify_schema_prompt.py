CLASSIFY_SCHEMA = """
You MUST output valid JSON with this structure:
Determine whether the user's request requires external tools.
Set "can_answer_confidence" to a value between 0 and 1 reflecting how confidently the provided context alone can answer the request. Do not use general knowledge.
If tools could help retrieve additional information, include the relevant tool categories.
{
  "applicable_tool_categories": ["<category1>", "<category2>"],
  "can_answer_confidence": 0.5
}
"""
