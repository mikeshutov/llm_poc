from __future__ import annotations

from typing import Any, Literal, Union

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from integrations.brave import BraveSearchClient
from integrations.brave.models import NewsSearchResponse, SuggestResponse, WebSearchResponse
from integrations.brave.search_type import SearchType
from integrations.brave.web_search_params import WebSearchParams


class GenericWebSearchArgs(BaseModel):
    query_text: str = Field(
        ...,
        description="Search query text. Use a single string.",
    )
    search_type: Literal["web_search", "news_search", "suggestion_search"] = Field(
        default="web_search",
        description="Type of web search to run.",
    )
    country: str = Field(
        default="CA",
        description="Two-letter country code used for search localization.",
    )
    count: int = Field(
        default=5,
        description="Maximum number of results to request.",
        ge=1,
    )
    params: dict[str, Any] | None = Field(
        default=None,
        description="Optional provider-specific parameters.",
    )


def _coerce_search_type(search_type: str) -> SearchType:
    try:
        return SearchType(search_type)
    except ValueError as exc:
        allowed = ", ".join(t.value for t in SearchType)
        raise ValueError(f"Invalid search_type '{search_type}'. Allowed values: {allowed}.") from exc


@tool(
    "generic_web_search",
    args_schema=GenericWebSearchArgs,
    description=f"""
Run a general web, news, or suggestion search.

Required fields:
- query_text (string)

Optional fields:
- search_type ({" | ".join(t.value for t in SearchType)})
- country (string)
- count (integer)
- params (object)

Example valid call:
{{
  "query_text": "best summer jackets",
  "search_type": "web_search"
}}
""",
)
def generic_web_search(
    query_text: str,
    search_type: str = "web_search",
    country: str = "CA",
    count: int = 5,
    params: dict[str, Any] | None = None,
) -> Union[WebSearchResponse, NewsSearchResponse, SuggestResponse]:
    brave_client = BraveSearchClient()
    match _coerce_search_type(search_type):
        case SearchType.NEWS_SEARCH:
            return brave_client.news_search(query_text)
        case SearchType.SUGGESTION_SEARCH:
            return brave_client.suggest(query_text)
        case _:
            return brave_client.web_search(
                WebSearchParams(q=query_text, country=country, count=count, extra_params=params or {})
            )
