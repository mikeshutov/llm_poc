from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from integrations.wikipedia import WikipediaClient
from integrations.wikipedia.models import WikipediaPageSummary, WikipediaSearchResult

_wikipedia_client = WikipediaClient()


class WikipediaSearchArgs(BaseModel):
    query: str = Field(
        ...,
        description="Wikipedia search query. Example: 'Toronto weather'.",
    )
    limit: int = Field(
        default=5,
        description="Maximum number of search results to return.",
        ge=1,
    )
    summary_sentences: int = Field(
        default=2,
        description="Number of sentences to include when fetching the top result summary.",
        ge=1,
    )


class WikipediaSearchResponse(BaseModel):
    query: str
    results: list[WikipediaSearchResult] = []
    top_result_summary: Optional[WikipediaPageSummary] = None


@tool(
    "wikipedia_search",
    args_schema=WikipediaSearchArgs,
    description="""
Search English Wikipedia and return matching pages plus a short summary of the top result.

Required fields:
- query (string)

Optional fields:
- limit (integer)
- summary_sentences (integer)

Example valid call:
{
  "query": "Toronto weather",
  "limit": 5,
  "summary_sentences": 2
}
""",
)
def wikipedia_search(
    query: str,
    limit: int = 5,
    summary_sentences: int = 2,
) -> WikipediaSearchResponse:
    results = _wikipedia_client.search(query, limit=limit)
    summary: Optional[WikipediaPageSummary] = None
    if results:
        try:
            summary = _wikipedia_client.get_page_summary(results[0].title, sentences=summary_sentences)
        except Exception:
            pass
    return WikipediaSearchResponse(query=query, results=results, top_result_summary=summary)
