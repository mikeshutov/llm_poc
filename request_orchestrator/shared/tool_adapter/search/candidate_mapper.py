from __future__ import annotations

from typing import Any

from integrations.brave.models import WebSearchResponse, WebSearchResult
from personalization.profile.models import UserProfile
from reranker import Candidate, rerank_candidates


def web_search_result_to_candidate(result: WebSearchResult) -> Candidate:
    return Candidate(
        id=result.url or result.title or "unknown-web-result",
        title=result.title or result.url,
        content={
            "name": result.title,
            "description": result.description,
            "url": result.url,
            "image_url": result.image_url,
        },
        metadata={
            "source": "web_search",
        },
    )


def rerank_web_search_response(
    response: WebSearchResponse,
    *,
    goal: str | None = None,
    user_profile: UserProfile | None = None,
    llm: Any | None = None,
    limit: int | None = None,
) -> WebSearchResponse:
    retrieved_count = len(response.results)
    if not response.results:
        return WebSearchResponse(
            query=response.query,
            results=[],
            retrieved_count=0,
            reranked=True,
        )

    candidates = [web_search_result_to_candidate(result) for result in response.results]
    ranked_candidates = rerank_candidates(
        candidates,
        goal=goal,
        user_profile=user_profile,
        llm=llm,
        limit=limit,
    )
    result_by_id = {
        (result.url or result.title or "unknown-web-result"): result
        for result in response.results
    }
    ranked_results: list[WebSearchResult] = []
    seen_ids: set[str] = set()

    for candidate in ranked_candidates:
        result = result_by_id.get(candidate.id)
        if result is None or candidate.id in seen_ids:
            continue
        ranked_results.append(result)
        seen_ids.add(candidate.id)

    if limit is not None:
        ranked_results = ranked_results[: max(1, limit)]

    return WebSearchResponse(
        query=response.query,
        results=ranked_results,
        retrieved_count=retrieved_count,
        reranked=True,
    )
