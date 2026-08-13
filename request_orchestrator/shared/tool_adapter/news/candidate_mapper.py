from __future__ import annotations

from typing import Any

from common.utils import normalize_text
from integrations.hn_algolia.models import HnHit, HnSearchResult
from request_orchestrator.shared.tool_adapter.news.constants import DEFAULT_HN_SEARCH_LIMIT
from reranker import Candidate, rerank_candidates


def hn_hit_to_candidate(hit: HnHit) -> Candidate:
    tag_text = ", ".join(tag for tag in (hit.tags or []) if tag)
    summary_parts = [
        cleaned
        for cleaned in (
            normalize_text(hit.author),
            f"{hit.points} points" if hit.points is not None else None,
            f"{hit.num_comments} comments" if hit.num_comments is not None else None,
            tag_text or None,
        )
        if cleaned is not None
    ]

    return Candidate(
        id=hit.object_id,
        title=normalize_text(hit.title) or hit.title or hit.url or hit.object_id,
        content={
            "name": normalize_text(hit.title) or hit.title,
            "summary": ". ".join(summary_parts) if summary_parts else None,
            "description": normalize_text(hit.story_text),
            "url": normalize_text(hit.url),
        },
        attributes={
            "author": normalize_text(hit.author),
            "tags": list(hit.tags or []),
        },
        metadata={
            "source": "hn_search",
            "created_at": hit.created_at,
        },
    )


def rerank_hn_search_result(
    response: HnSearchResult,
    *,
    goal: str | None = None,
    llm: Any | None = None,
) -> HnSearchResult:
    retrieved_count = len(response.hits)
    if not response.hits:
        return HnSearchResult(
            hits=[],
            nbHits=response.nb_hits,
            page=response.page,
            nbPages=response.nb_pages,
            retrieved_count=0,
            reranked=True,
        )

    candidates = [hn_hit_to_candidate(hit) for hit in response.hits]
    ranked_candidates = rerank_candidates(candidates, goal=goal, llm=llm)
    hit_by_id = {hit.object_id: hit for hit in response.hits}
    ranked_hits: list[HnHit] = []
    seen_ids: set[str] = set()

    for candidate in ranked_candidates:
        hit = hit_by_id.get(candidate.id)
        if hit is None or hit.object_id in seen_ids:
            continue
        ranked_hits.append(hit)
        seen_ids.add(hit.object_id)

    return HnSearchResult(
        hits=ranked_hits[:DEFAULT_HN_SEARCH_LIMIT],
        nbHits=response.nb_hits,
        page=response.page,
        nbPages=response.nb_pages,
        retrieved_count=retrieved_count,
        reranked=True,
    )
