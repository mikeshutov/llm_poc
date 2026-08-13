from __future__ import annotations

from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from requests.exceptions import RequestException

from integrations.hn_algolia import HnAlgoliaClient, HnSearchResult
from request_orchestrator.shared.tool_adapter.news.candidate_mapper import rerank_hn_search_result
from request_orchestrator.shared.tool_adapter.news.constants import DEFAULT_HN_SEARCH_LIMIT

_hn_client = HnAlgoliaClient()


class HnSearchArgs(BaseModel):
    query: str = Field(
        ...,
        description="Search query for Hacker News stories, comments, or discussions.",
    )
    sort_by: Literal["relevance", "date"] = Field(
        default="relevance",
        description="Sort results by 'relevance' (default) or 'date' for most recent first.",
    )


@tool(
    "hn_search",
    args_schema=HnSearchArgs,
    description="""
Search Hacker News stories and discussions via the Algolia API.

Required fields:
- query (string)

Optional fields:
- sort_by: 'relevance' (default) or 'date'

Returns story titles, URLs, authors, points, comment counts, and timestamps.

Example valid call:
{
  "query": "AI agents",
  "sort_by": "date"
}
""",
)
def hn_search(query: str, sort_by: str = "relevance") -> HnSearchResult | str:
    try:
        response = _hn_client.search(query, sort_by=sort_by, hits_per_page=DEFAULT_HN_SEARCH_LIMIT)
        return rerank_hn_search_result(response, goal=query)
    except RequestException as e:
        return f"Hacker News search unavailable: {e}"
