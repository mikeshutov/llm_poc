from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from integrations.brave import BraveSearchClient
from integrations.brave.models import NewsSearchResponse


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
    return BraveSearchClient().news_search(q)
