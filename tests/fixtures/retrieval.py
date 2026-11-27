from __future__ import annotations

import json

import pytest

from retrieval.reranker_models import Candidate, RerankerScore


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


@pytest.fixture
def reranker_factory():
    def create(*, ranked_ids):
        return FakeReranker(json.dumps({"ranked_ids": ranked_ids}))

    return create


class StaticBackend:
    def __init__(self, results):
        self.results = results
        self.query = None
        self.candidates = None

    def rerank(self, query, candidates):
        self.query = query
        self.candidates = candidates
        return self.results


@pytest.fixture
def static_reranker_factory():
    return StaticBackend
