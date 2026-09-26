from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from reranker import Candidate, RerankerClient, RerankerUnavailableError, build_candidate_evidence, build_reranker_query, rerank_candidates


class StaticBackend:
    def __init__(self, results):
        self.results = results
        self.query = None
        self.candidates = None

    def rerank(self, query, candidates):
        self.query = query
        self.candidates = candidates
        return self.results


def test_evidence_includes_semantic_fields_and_excludes_operational_fields() -> None:
    candidate = Candidate(
        id="p-1",
        title="Blue jacket",
        content={
            "name": "Blue jacket",
            "summary": "Lightweight shell",
            "description": "Water resistant",
            "text": "Outdoor jacket",
            "url": "https://example.com/p-1",
            "image_url": "https://example.com/p-1.jpg",
        },
        attributes={"color": "blue", "retrieval_distance": 0.1},
        metadata={"source": "db", "retrieval_distance": 0.1},
    )

    evidence = build_candidate_evidence(candidate)

    assert "Blue jacket" in evidence
    assert "Lightweight shell" in evidence
    assert "Water resistant" in evidence
    assert "Outdoor jacket" in evidence
    assert "color: blue" in evidence
    assert "example.com" not in evidence
    assert "retrieval_distance" not in evidence
    assert "source: db" not in evidence


def test_evidence_budget_is_deterministic() -> None:
    candidate = Candidate(id="1", title="Title", content={"description": "word " * 500})

    first = build_candidate_evidence(candidate, token_budget=64)
    second = build_candidate_evidence(candidate, token_budget=64)

    assert first == second
    assert len(first) <= 64 * 4


def test_query_composes_goal_and_profile_preferences() -> None:
    profile = Mock()
    profile.to_prompt_dict.return_value = {
        "user_attributes": {
            "attributes": [{"attribute_type": "style.preferences", "value": ["blue", "minimal"]}]
        }
    }

    query = build_reranker_query("winter jacket", profile)

    assert "query: winter jacket" in query
    assert "style.preferences" in query
    assert "blue" in query


def test_reranker_sorts_by_backend_scores_and_appends_unscored_candidates() -> None:
    backend = StaticBackend([
        {"id": "2", "score": 0.9},
        {"id": "unknown", "score": 1.0},
    ])
    candidates = [Candidate(id=str(index), title=f"Candidate {index}") for index in range(1, 13)]

    ranked = rerank_candidates(candidates, goal="best", backend=backend)

    assert [candidate.id for candidate in ranked] == ["2", "1", "3", "4", "5", "6", "7", "8", "9", "10"]
    assert backend.query == "query: best"
    assert [item["id"] for item in backend.candidates] == [str(index) for index in range(1, 13)]


def test_reranker_client_rejects_invalid_response() -> None:
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"results": [{"id": "1", "score": "bad"}]}

    with patch("reranker.client.requests.post", return_value=response):
        with pytest.raises(RerankerUnavailableError):
            RerankerClient(base_url="http://reranker").rerank("query", [{"id": "1", "text": "text"}])
