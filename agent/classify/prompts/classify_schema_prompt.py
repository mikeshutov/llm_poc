CLASSIFY_SCHEMA = """
You MUST output valid JSON with this structure:
Determine whether the user's request requires external tools.
Set "can_answer_without_tools" to true when provided context has answer do not use general knowledge.
If tools could help retrieve additional information, include the relevant tool categories.
Ensure that can_answer_without_tools is true only when confidence is above 0.8
{
  "applicable_tool_categories": ["<category1>", "<category2>"],
  "can_answer_without_tools": true,
  "confidence": 0.5
}
"""
