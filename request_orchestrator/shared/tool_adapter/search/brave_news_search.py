from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from integrations.brave import BraveSearchClient
from integrations.brave.models import NewsSearchResponse
from integrations.brave.search_type import SearchType
from request_orchestrator.models.evidence import EvidenceUrl, EvidenceUrlType, EvidenceView, ToolResult
from request_orchestrator.shared.tool_adapter.search.candidate_mapper import rerank_news_search_response
from reranker import DEFAULT_TOP_K
from tool.constants import TOOL_NAME_NEWS_SEARCH
from tool.constants import TOOL_RESULT_TYPE_NEWS_RESULTS


class NewsSearchArgs(BaseModel):
    q: str = Field(
        ...,
        description="News search query text.",
    )


class BraveNewsSearchMetadata(BaseModel):
    age: str | None = None


def _tool_result(result: NewsSearchResponse) -> ToolResult:
    evidence: list[EvidenceView] = []

    for news_item in result.results:
        url = (news_item.url or "").strip()
        metadata = BraveNewsSearchMetadata(age=news_item.age)
        hydrated = EvidenceView(
            item_id=url or (news_item.title or "").strip(),
            tool_name=TOOL_NAME_NEWS_SEARCH,
            title=(news_item.title or "").strip(),
            summary=(news_item.description or "").strip() or (news_item.age or "").strip(),
            urls=[EvidenceUrl(url=url, url_type=EvidenceUrlType.WEBSITE)] if url else [],
            image_url=(news_item.thumbnail_url or "").strip(),
            source=TOOL_NAME_NEWS_SEARCH,
            entity_type=TOOL_RESULT_TYPE_NEWS_RESULTS,
            llm_metadata=metadata.model_dump(exclude_none=True),
            raw_payload=news_item,
        )
        evidence.append(hydrated)

    return ToolResult(
        result=result,
        metadata={
            "query": result.query.model_dump(),
            "retrieved_count": result.retrieved_count,
            "reranked": result.reranked,
            "search_type": SearchType.NEWS_SEARCH.value,
        },

        evidence=evidence,
    )


@tool(
    TOOL_NAME_NEWS_SEARCH,
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
def news_search(q: str) -> ToolResult:
    response = BraveSearchClient().news_search(q)
    return _tool_result(rerank_news_search_response(response, goal=q, limit=DEFAULT_TOP_K))
