from __future__ import annotations

import json

from reranker.models import Candidate, RerankerScore


class FakeReranker:
    """Deterministic reranker backend for adapter tests."""

    def __init__(self, response: str):
        self.response = response
        self.last_query: str | None = None
        self.last_candidates: list[Candidate] | None = None

    def rerank(self, query: str, candidates: list[Candidate]) -> list[RerankerScore]:
        self.last_query = query
        self.last_candidates = candidates
        ranked_ids = json.loads(self.response).get("ranked_ids", [])
        return [
            RerankerScore(id=candidate_id, score=float(len(ranked_ids) - index))
            for index, candidate_id in enumerate(ranked_ids)
        ]
