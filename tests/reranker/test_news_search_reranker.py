from integrations.brave.models import NewsSearchResponse
from request_orchestrator.shared.tool_adapter.search.candidate_mapper import (
    news_result_to_candidate,
    rerank_news_search_response,
)
from test_utilities.fake_reranker import FakeReranker


def test_news_result_to_candidate_maps_result_fields() -> None:
    response = NewsSearchResponse.model_validate(
        {
            "query": {"original": "toronto news"},
            "results": [
                {
                    "title": "Toronto Transit Update",
                    "url": "https://example.com/transit",
                    "description": "Latest TTC service changes.",
                    "age": "2h",
                    "thumbnail_url": "https://example.com/transit.jpg",
                }
            ],
        }
    )

    candidate = news_result_to_candidate(response.results[0])

    assert candidate.id == "https://example.com/transit"
    assert candidate.fields == {
        "title": "Toronto Transit Update",
        "description": "Latest TTC service changes.",
        "published": "2h",
    }


def test_rerank_news_search_response_reorders_results_and_exposes_retrieval_metadata() -> None:
    llm = FakeReranker('{"ranked_ids": ["https://example.com/3", "https://example.com/2", "https://example.com/1", "https://example.com/4"]}')
    response = NewsSearchResponse.model_validate(
        {
            "query": {"original": "toronto news"},
            "results": [
                {"title": "One", "url": "https://example.com/1", "description": "desc 1"},
                {"title": "Two", "url": "https://example.com/2", "description": "desc 2"},
                {"title": "Three", "url": "https://example.com/3", "description": "desc 3"},
                {"title": "Four", "url": "https://example.com/4", "description": "desc 4"},
                {"title": "Five", "url": "https://example.com/5", "description": "desc 5"},
                {"title": "Six", "url": "https://example.com/6", "description": "desc 6"},
                {"title": "Seven", "url": "https://example.com/7", "description": "desc 7"},
            ],
        }
    )

    reranked = rerank_news_search_response(response, goal="find the most relevant news", llm=llm, limit=4)

    assert [result.url for result in reranked.results] == [
        "https://example.com/3",
        "https://example.com/2",
        "https://example.com/1",
        "https://example.com/4",
    ]
    assert reranked.retrieved_count == 7
    assert reranked.reranked is True


def test_rerank_news_search_response_skips_llm_when_result_count_is_at_or_below_limit() -> None:
    llm = FakeReranker('{"ranked_ids": ["https://example.com/2", "https://example.com/1"]}')
    response = NewsSearchResponse.model_validate(
        {
            "query": {"original": "toronto news"},
            "results": [
                {"title": "One", "url": "https://example.com/1", "description": "desc 1"},
                {"title": "Two", "url": "https://example.com/2", "description": "desc 2"},
            ],
        }
    )

    reranked = rerank_news_search_response(response, goal="find the most relevant news", llm=llm, limit=5)

    assert [result.url for result in reranked.results] == [
        "https://example.com/1",
        "https://example.com/2",
    ]
    assert reranked.retrieved_count == 2
    assert reranked.reranked is True
