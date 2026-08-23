from __future__ import annotations

from typing import Any, Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from integrations.brave import BraveSearchClient
from integrations.brave.models import NewsSearchResponse, WebSearchResponse
from integrations.brave.search_type import SearchType
from integrations.brave.web_search_params import WebSearchParams
from request_orchestrator.models.evidence import EvidenceUrl, EvidenceUrlType, EvidenceView, ToolResult
from request_orchestrator.shared.tool_adapter.search.candidate_mapper import (
    rerank_news_search_response,
    rerank_web_search_response,
)
from request_orchestrator.shared.tool_adapter.search.constants import DEFAULT_WEB_SEARCH_CANDIDATE_LIMIT
from reranker import DEFAULT_TOP_K
from tool.constants import TOOL_NAME_GENERIC_WEB_SEARCH
from tool.constants import TOOL_RESULT_TYPE_NEWS_RESULTS
from tool.constants import TOOL_RESULT_TYPE_WEB_SEARCH_RESULTS


class GenericWebSearchArgs(BaseModel):
    query_text: str = Field(
        ...,
        min_length=1,
        description="Search query text. Use a single string and do not leave it blank.",
    )
    search_type: Literal["web_search", "news_search", "suggestion_search"] = Field(
        default="web_search",
        description="Type of web search to run.",
    )
    country: str = Field(
        default="CA",
        description="Two-letter country code used for search localization.",
    )
    params: dict[str, Any] | None = Field(
        default=None,
        description="Optional provider-specific parameters.",
    )


class GenericNewsSearchMetadata(BaseModel):
    age: str | None = None


class GenericWebResultMetadata(BaseModel):
    pass


def _coerce_search_type(search_type: str) -> SearchType:
    try:
        return SearchType(search_type)
    except ValueError as exc:
        allowed = ", ".join(t.value for t in SearchType)
        raise ValueError(f"Invalid search_type '{search_type}'. Allowed values: {allowed}.") from exc


def _web_search_tool_result(result: WebSearchResponse) -> ToolResult:
    evidence: list[EvidenceView] = []
    for item in result.results:
        url = (item.url or "").strip()
        metadata = GenericWebResultMetadata()
        evidence_view = EvidenceView(
            item_id=url or item.title.strip(),
            tool_name=TOOL_NAME_GENERIC_WEB_SEARCH,
            title=item.title.strip(),
            summary=item.description.strip() or "Web search result.",
            urls=[EvidenceUrl(url=url, url_type=EvidenceUrlType.WEBSITE)] if url else [],
            image_url=(item.image_url or "").strip(),
            source=TOOL_NAME_GENERIC_WEB_SEARCH,
            entity_type=TOOL_RESULT_TYPE_WEB_SEARCH_RESULTS,
            llm_metadata=metadata.model_dump(exclude_none=True),
            raw_payload=item,
        )
        evidence.append(evidence_view)
    return ToolResult(
        result=result,
        metadata={
            "retrieved_count": result.retrieved_count,
            "reranked": result.reranked,
            "search_type": SearchType.WEB_SEARCH.value,
        },

        evidence=evidence,
    )


def _news_search_tool_result(result: NewsSearchResponse) -> ToolResult:
    evidence: list[EvidenceView] = []
    for item in result.results:
        url = (item.url or "").strip()
        metadata = GenericNewsSearchMetadata(age=item.age)
        evidence_view = EvidenceView(
            item_id=url or item.title.strip(),
            tool_name=TOOL_NAME_GENERIC_WEB_SEARCH,
            title=item.title.strip(),
            summary=item.description.strip() or (item.age or "").strip() or "News search result.",
            urls=[EvidenceUrl(url=url, url_type=EvidenceUrlType.WEBSITE)] if url else [],
            image_url=(item.thumbnail_url or "").strip(),
            source=TOOL_NAME_GENERIC_WEB_SEARCH,
            entity_type=TOOL_RESULT_TYPE_NEWS_RESULTS,
            llm_metadata=metadata.model_dump(exclude_none=True),
            raw_payload=item,
        )
        evidence.append(evidence_view)
    return ToolResult(
        result=result,
        metadata={
            "retrieved_count": result.retrieved_count,
            "reranked": result.reranked,
            "search_type": SearchType.NEWS_SEARCH.value,
        },

        evidence=evidence,
    )


@tool(
    TOOL_NAME_GENERIC_WEB_SEARCH,
    args_schema=GenericWebSearchArgs,
    description=f"""
Run a general web or news search. `suggestion_search` is treated the same as `web_search`.

Required fields:
- query_text (string)

Optional fields:
- search_type ({" | ".join(t.value for t in SearchType)})
- country (string)
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
    params: dict[str, Any] | None = None,
) -> ToolResult:
    normalized_query = query_text.strip()
    if not normalized_query:
        raise ValueError("query_text is required and cannot be blank.")

    brave_client = BraveSearchClient()
    match _coerce_search_type(search_type):
        case SearchType.NEWS_SEARCH:
            response = brave_client.news_search(normalized_query)
            return _news_search_tool_result(rerank_news_search_response(response, goal=normalized_query, limit=DEFAULT_TOP_K))
        #case SearchType.SUGGESTION_SEARCH:
        #    return brave_client.suggest(query_text)
        case _:
            response = brave_client.web_search(
                WebSearchParams(
                    q=normalized_query,
                    country=country,
                    count=DEFAULT_WEB_SEARCH_CANDIDATE_LIMIT,
                    extra_params=params or {},
                )
            )
            return _web_search_tool_result(rerank_web_search_response(response, goal=normalized_query, limit=DEFAULT_TOP_K))
