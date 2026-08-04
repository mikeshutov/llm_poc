from __future__ import annotations

from typing import Literal, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from requests.exceptions import RequestException

from integrations.hn_algolia import HnAlgoliaClient, HnSearchResult

_hn_client = HnAlgoliaClient()


class HnSearchArgs(BaseModel):
    query: str = Field(
        ...,
        description="Search query for Hacker News stories, comments, or discussions.",
    )
    sort_by: Optional[Literal["relevance", "date"]] = Field(
        default="relevance",
        description="Sort results by 'relevance' (default) or 'date' for most recent first.",
    )
    limit: Optional[int] = Field(
        default=10,
        description="Number of results to return (1–50). Defaults to 10.",
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
- limit (integer, default 10)

Returns story titles, URLs, authors, points, comment counts, and timestamps.

Example valid call:
{
  "query": "AI agents",
  "sort_by": "date"
}
""",
)
def hn_search(query: str, sort_by: str = "relevance", limit: int = 10) -> HnSearchResult | str:
    try:
        return _hn_client.search(query, sort_by=sort_by, hits_per_page=limit)
    except RequestException as e:
        return f"Hacker News search unavailable: {e}"
