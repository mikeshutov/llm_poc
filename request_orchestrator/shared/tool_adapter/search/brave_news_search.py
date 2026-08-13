from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from integrations.brave import BraveSearchClient
from integrations.brave.models import NewsSearchResponse
from request_orchestrator.shared.tool_adapter.search.candidate_mapper import rerank_news_search_response
from reranker import DEFAULT_TOP_K


class NewsSearchArgs(BaseModel):
    q: str = Field(
        ...,
        description="News search query text.",
    )


@tool(
    "news_search",
    args_schema=NewsSearchArgs,
    description="""
Search for current news results using Brave News Search.

Required fields:
- q (string)

Example valid call:
{
  "q": "Toronto weather clothing news"
}
""",
)
def news_search(q: str) -> NewsSearchResponse:
    response = BraveSearchClient().news_search(q)
    return rerank_news_search_response(response, goal=q, limit=DEFAULT_TOP_K)
