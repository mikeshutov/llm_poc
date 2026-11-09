from __future__ import annotations

from pathlib import Path


RESEARCH_TEMPLATE_PATH = Path(__file__).with_name("research-template.md")


def load_research_template() -> str:
    return RESEARCH_TEMPLATE_PATH.read_text(encoding="utf-8").strip()


def build_research_rules() -> str:
    return (
        "Research planning rules:\n"
        "- Break the topic into the smallest useful research questions before choosing tools.\n"
        "- Prefer authoritative or primary sources when available, then corroborate important claims.\n"
        "- Use web search for fresh or broad information, web fetch for details from a promising page, "
        "and knowledge tools for stable facts.\n"
        "- For the web_search tool, pass the required argument `query_text` (not `query`), along "
        "with `search_type` when needed and a two-letter `country` code; follow the tool schema "
        "exactly and do not retry with an invalid argument name.\n"
        "- Use news search when the request concerns recent events and books search for published works.\n"
        "- Avoid duplicate searches; refine or broaden a query only when the evidence is insufficient.\n"
        "- Track uncertainty, source quality, publication dates, and disagreements between sources.\n"
        "- Use the supplied research template to structure the final report.\n"
        "- After gathering and checking the evidence, write the completed report to the conversation "
        "workspace as `research-{topic}.md`, using a concise kebab-case topic slug.\n"
        "- Treat the completed `research-{topic}.md` workspace file as the definitive research output. "
        "Do not finish with only a collection of search results.\n"
        "- After the report is successfully written, do not continue searching unless writing or "
        "validation failed.\n"
        "- Use workspace_file_read or workspace_file_list to validate or revise the report, then "
        "call workspace_file_complete with the final filename as the last tool call. Do not "
        "promote it to managed artifacts.\n"
        "- Do not answer the user directly; gather evidence for the parent agent.\n"
        "- Keep each plan step explicit, focused, and independently executable."
    )


__all__ = [
    "build_research_rules",
    "load_research_template",
]
