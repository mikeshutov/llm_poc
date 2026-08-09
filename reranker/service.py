from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI

from common.parsing import strip_code_fences
from conversation.models.conversation_model_config import ConversationModelConfig, RERANKER_STAGE, SHARED_MODEL_SCOPE
from personalization.profile.models import UserProfile
from request_orchestrator.shared.runtime_context import get_current_conversation_model_config
from reranker.constants import DEFAULT_TOP_K
from reranker.models import Candidate, RerankerPrompt, RerankerResult


class CandidateReranker:
    def __init__(
        self,
        llm: Any | None = None,
        conversation_model_config: ConversationModelConfig | None = None,
    ):
        resolved_config = conversation_model_config or get_current_conversation_model_config() or ConversationModelConfig.build_default()
        resolved_model = resolved_config.resolve(SHARED_MODEL_SCOPE, RERANKER_STAGE)
        self.llm = ChatOpenAI(model=resolved_model) if llm is None else llm

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
            return list(candidates)[:resolved_limit]

        candidate_by_id = {candidate.id: candidate for candidate in candidates}
        ranked_candidates = self._sort_candidates(candidate_by_id, candidates, rerank_result.ranked_ids)
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
    conversation_model_config: ConversationModelConfig | None = None,
) -> list[Candidate]:
    return CandidateReranker(llm=llm, conversation_model_config=conversation_model_config).rerank(
        candidates,
        goal=goal,
        query=query,
        user_profile=user_profile,
        limit=limit,
    )
