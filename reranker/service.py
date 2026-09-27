from __future__ import annotations

from typing import Any

from personalization.profile.models import UserProfile
from reranker.client import RerankerBackend, RerankerClient
from reranker.constants import DEFAULT_TOP_K
from reranker.models import Candidate, RerankerScore
from reranker.query import build_reranker_query


class CandidateReranker:
    def __init__(
        self,
        llm: Any | None = None,
        backend: RerankerBackend | None = None,
    ):
        self.backend = backend or (llm if hasattr(llm, "rerank") else RerankerClient())

    def rerank(
        self,
        candidates: list[Candidate],
        *,
        goal: str | None = None,
        query: str | None = None,
        user_profile: UserProfile | None = None,
        limit: int | None = None,
    ) -> list[Candidate]:
        resolved_limit = DEFAULT_TOP_K if limit is None else max(1, limit)

        if len(candidates) <= resolved_limit:
            return list(candidates)[:resolved_limit]

        resolved_goal = goal if goal is not None else query

        query_text = build_reranker_query(resolved_goal, user_profile)
        evidence = candidates
        try:
            scored = [
                item if isinstance(item, RerankerScore) else RerankerScore.model_validate(item)
                for item in self.backend.rerank(query_text, evidence)
            ]
        except Exception as exc:
            raise
        candidate_by_id = {candidate.id: candidate for candidate in candidates}
        ranked_candidates = self._sort_candidates(
            candidate_by_id,
            candidates,
            [item.id for item in scored],
        )
        return ranked_candidates[:resolved_limit]

    def _sort_candidates(
        self,
        candidate_by_id: dict[str, Candidate],
        candidates: list[Candidate],
        ranked_candidate_ids: list[str],
    ) -> list[Candidate]:
        seen_ids: set[str] = set()
        ranked_candidates: list[Candidate] = []

        for candidate_id in ranked_candidate_ids:
            if candidate_id in seen_ids:
                continue
            candidate = candidate_by_id.get(candidate_id)
            if candidate is None:
                continue
            ranked_candidates.append(candidate)
            seen_ids.add(candidate_id)

        for candidate in candidates:
            if candidate.id in seen_ids:
                continue
            ranked_candidates.append(candidate)

        return ranked_candidates


def rerank_candidates(
    candidates: list[Candidate],
    *,
    goal: str | None = None,
    query: str | None = None,
    user_profile: UserProfile | None = None,
    llm: Any | None = None,
    limit: int | None = None,
    backend: RerankerBackend | None = None,
) -> list[Candidate]:
    return CandidateReranker(
        llm=llm,
        backend=backend,
    ).rerank(
        candidates,
        goal=goal,
        query=query,
        user_profile=user_profile,
        limit=limit,
    )
