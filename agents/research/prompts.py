from __future__ import annotations

def build_research_rules() -> str:
    return (
        "Research-specific rules:\n"
        "- Prefer authoritative or primary sources when available, then corroborate important claims.\n"
        "- Use web search for fresh or broad information, web fetch for details from a promising page, "
        "and knowledge tools for stable facts.\n"
        "- For the web_search tool, pass the required argument `query_text` (not `query`), along "
        "with `search_type` when needed and a two-letter `country` code; follow the tool schema "
        "exactly and do not retry with an invalid argument name.\n"
        "- Use news search when the request concerns recent events and books search for published works.\n"
        "- Avoid duplicate searches; refine or broaden a query only when the evidence is insufficient.\n"
        "- Track uncertainty, source quality, publication dates, and disagreements between sources.\n"
        "- Record the checked research in working memory as the complete handoff for a downstream "
        "document-writing agent. Include source titles, publication or access dates, URLs, "
        "quantitative facts, evidence IDs, source quality, uncertainty, disagreements, and "
        "limitations needed to write the final document.\n"
        "- Do not create, write, validate, or promote a separate research report file. The "
        "downstream document-writing agent is responsible for converting the working-memory "
        "handoff into the requested document format.\n"
        "- After the research findings and remaining gaps are recorded, make a final successful "
        "update_memory_state with task_completed=true; that update is the completion signal."
    )


__all__ = [
    "build_research_rules",
]
