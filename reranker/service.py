from __future__ import annotations

from time import perf_counter
from typing import Any

from common.data import sanitize_for_json_storage
from common.logging import create_conversation_event
from personalization.profile.models import UserProfile
from reranker.client import RerankerBackend, RerankerClient
from reranker.constants import DEFAULT_TOP_K, RERANKER_EVENT_TYPE, RERANKER_MODEL_NAME
from reranker.models import Candidate, RerankerScore
from reranker.query import build_reranker_query
from request_orchestrator.shared.runtime_context import get_current_agent_name


def _candidate_log_payload(
    candidates: list[Candidate],
    backend: RerankerBackend,
) -> tuple[list[dict[str, Any]], list[int]]:
    if isinstance(backend, RerankerClient):
        prepared = backend.prepare_candidates(candidates)
        payload = [candidate.model_dump(mode="json") for candidate in prepared]
        return payload, [len(candidate.text) for candidate in prepared]

    payload = [
        {
            "id": candidate.id,
            "fields": sanitize_for_json_storage(candidate.fields),
        }
        for candidate in candidates
    ]
    return (
        payload,
        [len(str(candidate.fields)) for candidate in candidates],
    )


def _record_reranker_event(
    *,
    query: str,
    candidates: list[Candidate],
    backend: RerankerBackend,
    results: list[RerankerScore] | None = None,
    limit: int,
    latency_ms: int,
    batch_size: int | None,
    error: str = "",
) -> None:
    candidate_payload, evidence_lengths = _candidate_log_payload(candidates, backend)
    create_conversation_event(
        event_type=RERANKER_EVENT_TYPE,
        source="reranker",
        agent_name=get_current_agent_name() or "",
        node_name="reranker",
        payload={
            "kind": RERANKER_EVENT_TYPE,
            "title": "Reranker",
            "model": RERANKER_MODEL_NAME,
            "query": query,
            "candidates": candidate_payload,
            "results": [score.model_dump(mode="json") for score in (results or [])],
            "candidate_count": len(candidates),
            "limit": limit,
            "evidence_lengths": evidence_lengths,
            "batch_size": batch_size,
            "latency_ms": latency_ms,
            "error": error,
        },
    )


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
        started_at = perf_counter()
        scored: list[RerankerScore] = []
        error = ""
        try:
            scored = [
                item if isinstance(item, RerankerScore) else RerankerScore.model_validate(item)
                for item in self.backend.rerank(query_text, evidence)
            ]
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            _record_reranker_event(
                query=query_text,
                candidates=evidence,
                backend=self.backend,
                results=scored,
                limit=resolved_limit,
                latency_ms=int((perf_counter() - started_at) * 1000),
                batch_size=getattr(self.backend, "batch_size", None),
                error=error,
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
