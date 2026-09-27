from unittest.mock import Mock, patch

import pytest

from reranker import RerankerCandidate, RerankerClient, RerankerUnavailableError, build_reranker_query, rerank_candidates


class StaticBackend:
    def __init__(self, results):
        self.results = results
        self.query = None
        self.candidates = None

    def rerank(self, query, candidates):
        self.query = query
        self.candidates = candidates
        return self.results


def test_reranker_sorts_by_backend_scores_and_appends_unscored_candidates() -> None:
    backend = StaticBackend([
        {"id": "2", "score": 0.9},
        {"id": "unknown", "score": 1.0},
    ])
    candidates = [RerankerCandidate(id=str(index), text=f"Candidate {index}") for index in range(1, 13)]

    ranked = rerank_candidates(candidates, goal="best", backend=backend)

    assert [candidate.id for candidate in ranked] == ["2", "1", "3", "4", "5", "6", "7", "8", "9", "10"]
    assert backend.query == "query: best"
    assert [item.id for item in backend.candidates] == [str(index) for index in range(1, 13)]


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


def test_reranker_client_applies_candidate_budget_before_serialization() -> None:
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"results": [{"id": "1", "score": 0.5}]}

    with patch("reranker.client.requests.post", return_value=response) as post:
        RerankerClient(base_url="http://reranker").rerank(
            "query",
            [RerankerCandidate(id="1", text="word " * 500)],
        )

    submitted = post.call_args.kwargs["json"]["candidates"][0]["text"]
    assert len(submitted) <= 384 * 4


def test_reranker_client_rejects_invalid_response() -> None:
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"results": [{"id": "1", "score": "bad"}]}

    with patch("reranker.client.requests.post", return_value=response):
        with pytest.raises(RerankerUnavailableError):
            RerankerClient(base_url="http://reranker").rerank(
                "query",
                [RerankerCandidate(id="1", text="text")],
            )
