from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI

from common.model_constants import RERANKER_MODEL
from common.parsing import strip_code_fences
from personalization.profile.models import UserProfile
from reranker.constants import DEFAULT_TOP_K
from reranker.models import Candidate, RerankerPrompt, RerankerResult


class CandidateReranker:
    def __init__(self, llm: Any | None = None):
        self.llm = ChatOpenAI(model=RERANKER_MODEL) if llm is None else llm

    def rerank(
        self,
        candidates: list[Candidate],
        *,
        goal: str | None = None,
        query: str | None = None,
        user_profile: UserProfile | None = None,
    ) -> list[Candidate]:
        if len(candidates) <= DEFAULT_TOP_K:
            return list(candidates)[:DEFAULT_TOP_K]

        resolved_goal = goal if goal is not None else query

        prompt = RerankerPrompt(
            goal=resolved_goal or "",
            user_profile=user_profile,
            candidates=candidates,
        ).to_prompt_text()
        raw = self.llm.invoke(prompt).content
        raw = strip_code_fences(raw)

        try:
            rerank_result = RerankerResult.model_validate_json(raw)
        except Exception:
            return list(candidates)[:DEFAULT_TOP_K]

        candidate_by_id = {candidate.id: candidate for candidate in candidates}
        ranked_candidates = self._sort_candidates(candidate_by_id, candidates, rerank_result.ranked_ids)
        return ranked_candidates[:DEFAULT_TOP_K]

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
) -> list[Candidate]:
    return CandidateReranker(llm=llm).rerank(candidates, goal=goal, query=query, user_profile=user_profile)
