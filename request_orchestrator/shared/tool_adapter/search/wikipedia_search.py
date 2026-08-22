from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from integrations.wikipedia import WikipediaClient
from integrations.wikipedia.models import WikipediaPageSummary, WikipediaSearchResult
from request_orchestrator.models.evidence import EvidenceUrl, EvidenceView, HydratedEvidence, ToolResult
from tool.constants import TOOL_NAME_WIKIPEDIA_SEARCH
from tool.constants import TOOL_RESULT_TYPE_KNOWLEDGE

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


class WikipediaSearchMetadata(BaseModel):
    top_result_summary: dict[str, object] | None = None


def _tool_result(result: WikipediaSearchResponse) -> ToolResult:
    hydrated_evidence: list[HydratedEvidence] = []
    evidence_views: list[EvidenceView] = []
    if not result.results and result.top_result_summary is not None:
        summary_url = result.top_result_summary.url.strip()
        metadata = WikipediaSearchMetadata(
            top_result_summary=result.top_result_summary.model_dump(),
        )
        hydrated = HydratedEvidence(
            item_id=summary_url or result.top_result_summary.title.strip(),
            tool_name=TOOL_NAME_WIKIPEDIA_SEARCH,
            title=result.top_result_summary.title.strip(),
            summary=result.top_result_summary.summary.strip() or "Wikipedia result.",
            urls=[EvidenceUrl(url=summary_url, url_type="website")] if summary_url else [],
            source=TOOL_NAME_WIKIPEDIA_SEARCH,
            entity_type=TOOL_RESULT_TYPE_KNOWLEDGE,
            metadata=metadata.model_dump(exclude_none=True),
            raw_payload=result.top_result_summary,
        )
        return ToolResult(
            result=result,
            evidence_views=[
                EvidenceView(
                    item_id=hydrated.item_id,
                    title=hydrated.title,
                    summary=hydrated.summary,
                    metadata=dict(hydrated.metadata),
                )
            ],
            hydrated_evidence=[hydrated],
        )

    for index, item in enumerate(result.results):
        summary_text = item.description.strip()
        if index == 0 and result.top_result_summary is not None and result.top_result_summary.summary.strip():
            summary_text = result.top_result_summary.summary.strip()
        url = item.url.strip()
        metadata = WikipediaSearchMetadata(
            top_result_summary=(
                result.top_result_summary.model_dump()
                if index == 0 and result.top_result_summary is not None
                else None
            ),
        )
        hydrated = HydratedEvidence(
            item_id=url or item.title.strip(),
            tool_name=TOOL_NAME_WIKIPEDIA_SEARCH,
            title=item.title.strip(),
            summary=summary_text or "Wikipedia result.",
            urls=[EvidenceUrl(url=url, url_type="website")] if url else [],
            source=TOOL_NAME_WIKIPEDIA_SEARCH,
            entity_type=TOOL_RESULT_TYPE_KNOWLEDGE,
            metadata=metadata.model_dump(exclude_none=True),
            raw_payload=item,
        )
        hydrated_evidence.append(hydrated)
        evidence_views.append(
            EvidenceView(
                item_id=hydrated.item_id,
                title=hydrated.title,
                summary=hydrated.summary,
                metadata=dict(hydrated.metadata),
            )
        )
    return ToolResult(result=result, evidence_views=evidence_views, hydrated_evidence=hydrated_evidence)


@tool(
    TOOL_NAME_WIKIPEDIA_SEARCH,
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
) -> ToolResult:
    results = _wikipedia_client.search(query, limit=limit)
    summary: Optional[WikipediaPageSummary] = None
    if results:
        try:
            summary = _wikipedia_client.get_page_summary(results[0].title, sentences=summary_sentences)
        except Exception:
            pass
    return _tool_result(WikipediaSearchResponse(query=query, results=results, top_result_summary=summary))
