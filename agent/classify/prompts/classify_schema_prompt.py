CLASSIFY_SCHEMA = """
You MUST output valid JSON with this structure:
Determine whether the user's request requires external tools.
Set "can_answer_without_tools" to true only if the user's request can be answered well using the conversation context already provided, without retrieving additional external information.
If tools could help retrieve additional information, include the relevant tool categories.
{
  "applicable_tool_categories": ["<category1>", "<category2>"],
  "can_answer_without_tools": true,
  "confidence": 0.5
}
"""
