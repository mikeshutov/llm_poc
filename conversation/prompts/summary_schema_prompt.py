SUMMARY_SCHEMA_PROMPT = """
Return a JSON object with two fields:
- "conversation_summary": a concise factual summary as a single string preserving user preferences, constraints, and decisions made.
- "tool_summary": a concise summary as a single string of tools used, what was looked up, and the key results. Focus on data that may be useful in future turns (e.g. prices, locations, facts retrieved).

Return only valid JSON. Example:
{
    "conversation_summary": "User is looking for running shoes under $100 in size 10...",
    "tool_summary": "find_products: returned 3 Nike shoes. exchange_rates_lookup: EUR/USD was 1.08 on March 15th."
}
"""
