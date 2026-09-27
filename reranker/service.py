from __future__ import annotations

from time import perf_counter
from typing import Any

from common.logging import create_conversation_event
from personalization.profile.models import UserProfile
from request_orchestrator.shared.runtime_context import (
    get_current_agent_name,
    get_current_roundtrip_id,
)
from reranker.client import RerankerBackend, RerankerClient
from reranker.constants import DEFAULT_TOP_K
from reranker.models import RerankerCandidate, RerankerScore
from reranker.query import build_reranker_query


class CandidateReranker:
    def __init__(
        self,
        llm: Any | None = None,
        conversation_model_config: Any | None = None,
        backend: RerankerBackend | None = None,
    ):
        self.backend = backend or (llm if hasattr(llm, "rerank") else RerankerClient())
        self.model_name = "BAAI/bge-reranker-v2-m3"

    def rerank(
        self,
        candidates: list[RerankerCandidate],
        *,
        goal: str | None = None,
        query: str | None = None,
        user_profile: UserProfile | None = None,
        limit: int | None = None,
    ) -> list[RerankerCandidate]:
        resolved_limit = DEFAULT_TOP_K if limit is None else max(1, limit)

        if len(candidates) <= resolved_limit:
            return list(candidates)[:resolved_limit]

        resolved_goal = goal if goal is not None else query

        query_text = build_reranker_query(resolved_goal, user_profile)
        evidence = list(candidates)
        started_at = perf_counter()
        try:
            scored = [
                item if isinstance(item, RerankerScore) else RerankerScore.model_validate(item)
                for item in self.backend.rerank(query_text, evidence)
            ]
        except Exception as exc:
            create_conversation_event(
                event_type="reranker_call",
                source="reranker.candidate_reranker",
                agent_name=get_current_agent_name() or "",
                node_name="reranker",
                payload={
                    "model": self.model_name,
                    "candidate_count": len(candidates),
                    "limit": resolved_limit,
                    "query": query_text,
                    "candidates": [item.model_dump() for item in evidence],
                    "results": [],
                    "error": str(exc),
                    "roundtrip_id": get_current_roundtrip_id(),
                },
            )
            raise
        latency_ms = int((perf_counter() - started_at) * 1000)
        create_conversation_event(
            event_type="reranker_call",
            source="reranker.candidate_reranker",
            agent_name=get_current_agent_name() or "",
            node_name="reranker",
            payload={
                "model": self.model_name,
                "candidate_count": len(candidates),
                "limit": resolved_limit,
                "query": query_text,
                "candidates": [item.model_dump() for item in evidence],
                "results": [item.model_dump() for item in scored],
                "evidence_lengths": [len(item.text) for item in evidence],
                "batch_size": getattr(self.backend, "batch_size", None),
                "latency_ms": latency_ms,
                "roundtrip_id": get_current_roundtrip_id(),
            },
        )
        candidate_by_id = {candidate.id: candidate for candidate in candidates}
        ranked_candidates = self._sort_candidates(
            candidate_by_id,
            candidates,
            [item.id for item in scored],
        )
        return ranked_candidates[:resolved_limit]

    def _sort_candidates(
        self,
        candidate_by_id: dict[str, RerankerCandidate],
        candidates: list[RerankerCandidate],
        ranked_candidate_ids: list[str],
    ) -> list[RerankerCandidate]:
        seen_ids: set[str] = set()
        ranked_candidates: list[RerankerCandidate] = []

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
    candidates: list[RerankerCandidate],
    *,
    goal: str | None = None,
    query: str | None = None,
    user_profile: UserProfile | None = None,
    llm: Any | None = None,
    limit: int | None = None,
    conversation_model_config: Any | None = None,
    backend: RerankerBackend | None = None,
) -> list[RerankerCandidate]:
    return CandidateReranker(
        llm=llm,
        conversation_model_config=conversation_model_config,
        backend=backend,
    ).rerank(
        candidates,
        goal=goal,
        query=query,
        user_profile=user_profile,
        limit=limit,
    )
