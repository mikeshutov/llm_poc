from integrations.hn_algolia.models import HnSearchResult
from request_orchestrator.shared.tool_adapter.news.candidate_mapper import (
    hn_hit_to_candidate,
    rerank_hn_search_result,
)
from request_orchestrator.shared.tool_adapter.news.constants import DEFAULT_HN_SEARCH_LIMIT
from test_utilities.mock_llm import MockLLM


def test_hn_hit_to_candidate_maps_hit_fields() -> None:
    response = HnSearchResult.model_validate(
        {
            "nbHits": 1,
            "hits": [
                {
                    "objectID": "1",
                    "title": "  AI Agents Are Getting Better  ",
                    "url": " https://example.com/ai-agents ",
                    "author": "  pg  ",
                    "points": 123,
                    "num_comments": 45,
                    "created_at": "2026-08-13T10:00:00Z",
                    "story_text": "  Discussion of agent progress.  ",
                    "_tags": ["story", "ai"],
                }
            ],
        }
    )

    candidate = hn_hit_to_candidate(response.hits[0])

    assert candidate.id == "1"
    assert candidate.title == "AI Agents Are Getting Better"
    assert candidate.content["name"] == "AI Agents Are Getting Better"
    assert candidate.content["summary"] == "pg. 123 points. 45 comments. story, ai"
    assert candidate.content["description"] == "Discussion of agent progress."
    assert candidate.content["url"] == "https://example.com/ai-agents"
    assert candidate.metadata["source"] == "hn_search"


def test_rerank_hn_search_result_reorders_hits_and_preserves_metadata() -> None:
    llm = MockLLM([
        '{"ranked_ids": ["7", "6", "5", "4", "3", "2"]}'
    ])
    response = HnSearchResult.model_validate(
        {
            "nbHits": 7,
            "hits": [
                {"objectID": "1", "title": "One"},
                {"objectID": "2", "title": "Two"},
                {"objectID": "3", "title": "Three"},
                {"objectID": "4", "title": "Four"},
                {"objectID": "5", "title": "Five"},
                {"objectID": "6", "title": "Six"},
                {"objectID": "7", "title": "Seven"},
            ],
        }
    )

    reranked = rerank_hn_search_result(response, goal="best ai discussions", llm=llm)

    assert [hit.object_id for hit in reranked.hits] == ["7", "6", "5", "4", "3", "2"]
    assert reranked.retrieved_count == 7
    assert reranked.reranked is True


def test_rerank_hn_search_result_skips_llm_when_result_count_is_at_or_below_limit() -> None:
    llm = MockLLM([
        '{"ranked_ids": ["2", "1"]}'
    ])
    response = HnSearchResult.model_validate(
        {
            "nbHits": 2,
            "hits": [
                {"objectID": "1", "title": "One"},
                {"objectID": "2", "title": "Two"},
            ],
        }
    )

    reranked = rerank_hn_search_result(response, goal="best ai discussions", llm=llm)

    assert [hit.object_id for hit in reranked.hits] == ["1", "2"]
    assert reranked.retrieved_count == 2
    assert reranked.reranked is True
    assert llm.last_prompt is None
    assert DEFAULT_HN_SEARCH_LIMIT == 20
