SUMMARY_SCHEMA_PROMPT = """
Return a JSON object with one field:
- conversation_summary: returns a concise factual rolling summary as a single string made of bullet points preserving user preferences, constraints, and decisions made.

Return only valid JSON. Example:
{
    "conversation_summary": "- User is looking for running shoes under $100 in size 10\n- User prefers neutral colors\n- Compared two candidate products and rejected one for price"
}
"""
