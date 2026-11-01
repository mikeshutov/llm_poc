from __future__ import annotations


def build_research_rules() -> str:
    return (
        "Research planning rules:\n"
        "- Break the topic into the smallest useful research questions before choosing tools.\n"
        "- Prefer authoritative or primary sources when available, then corroborate important claims.\n"
        "- Use web search for fresh or broad information, web fetch for details from a promising page, "
        "and knowledge tools for stable facts.\n"
        "- Use news search when the request concerns recent events and books search for published works.\n"
        "- Avoid duplicate searches; refine or broaden a query only when the evidence is insufficient.\n"
        "- Track uncertainty, source quality, publication dates, and disagreements between sources.\n"
        "- Return a collection of evidence records for downstream agents; evidence IDs can be "
        "looked up later to retrieve the full records.\n"
        "- Do not mutate files, user profiles, or other durable state.\n"
        "- Do not answer the user directly; gather evidence for the parent agent.\n"
        "- Keep each plan step explicit, focused, and independently executable."
    )


__all__ = ["build_research_rules"]
