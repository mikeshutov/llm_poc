from __future__ import annotations

from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from requests.exceptions import RequestException

from integrations.hn_algolia import HnAlgoliaClient
from integrations.hn_algolia.models import HnHit, HnSearchResult
from request_orchestrator.models.evidence import EvidenceUrl, EvidenceUrlType, EvidenceView, ToolResult
from request_orchestrator.shared.tool_adapter.news.candidate_mapper import rerank_hn_search_result
from request_orchestrator.shared.tool_adapter.news.constants import DEFAULT_HN_SEARCH_LIMIT
from tool.constants import TOOL_NAME_HN_SEARCH
from tool.constants import TOOL_RESULT_TYPE_NEWS_RESULTS

_hn_client = HnAlgoliaClient()


class HnSearchMetadata(BaseModel):
    author: str | None = None
    points: int | None = None
    num_comments: int | None = None
    tags: list[str] = []


def _hit_summary(hit: HnHit) -> str:
    parts: list[str] = []
    if hit.author:
        parts.append(hit.author)
    if hit.points is not None:
        parts.append(f"{hit.points} points")
    if hit.num_comments is not None:
        parts.append(f"{hit.num_comments} comments")
    if hit.story_text:
        parts.append(hit.story_text)
    return ". ".join(parts) if parts else "Hacker News result."


def _tool_result(result: HnSearchResult) -> ToolResult:
    evidence: list[EvidenceView] = []
    for hit in result.hits:
        url = (hit.url or "").strip()
        metadata = HnSearchMetadata(
            author=hit.author,
            points=hit.points,
            num_comments=hit.num_comments,
            tags=list(hit.tags or []),
        )
        evidence_view = EvidenceView(
            item_id=hit.object_id,
            tool_name=TOOL_NAME_HN_SEARCH,
            title=(hit.title or hit.url or hit.object_id or "").strip(),
            summary=_hit_summary(hit),
            urls=[EvidenceUrl(url=url, url_type=EvidenceUrlType.WEBSITE)] if url else [],
            published_at=(hit.created_at or "").strip(),
            source=TOOL_NAME_HN_SEARCH,
            entity_type=TOOL_RESULT_TYPE_NEWS_RESULTS,
            llm_metadata=metadata.model_dump(exclude_none=True),
            raw_payload=hit,
        )
        evidence.append(evidence_view)
    return ToolResult(result=result, evidence=evidence)




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
    TOOL_NAME_HN_SEARCH,
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
def hn_search(query: str, sort_by: str = "relevance") -> ToolResult:
    try:
        response = _hn_client.search(query, sort_by=sort_by, hits_per_page=DEFAULT_HN_SEARCH_LIMIT)
        return _tool_result(rerank_hn_search_result(response, goal=query))
    except RequestException as e:
        return ToolResult.error(f"Hacker News search unavailable: {e}")
