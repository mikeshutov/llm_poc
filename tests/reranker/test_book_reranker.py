from integrations.open_library.models import BookSearchResult
from request_orchestrator.shared.tool_adapter.books.candidate_mapper import (
    book_to_candidate,
    rerank_book_search_result,
)
from request_orchestrator.shared.tool_adapter.books.constants import DEFAULT_BOOK_SEARCH_LIMIT
from test_utilities.mock_llm import MockLLM


def test_book_to_candidate_maps_book_fields() -> None:
    response = BookSearchResult.model_validate(
        {
            "numFound": 1,
            "docs": [
                {
                    "key": "/works/OL1W",
                    "title": "  The Hobbit  ",
                    "author_name": [" J.R.R. Tolkien "],
                    "first_publish_year": 1937,
                    "edition_count": 12,
                    "subject": [" Fantasy ", " Adventure "],
                    "publisher": [" Allen & Unwin "],
                    "language": ["eng"],
                    "cover_i": 12345,
                }
            ],
        }
    )

    candidate = book_to_candidate(response.docs[0])

    assert candidate.id == "/works/OL1W"
    assert candidate.title == "The Hobbit"
    assert candidate.content["name"] == "The Hobbit"
    assert candidate.content["summary"] == "J.R.R. Tolkien. 1937. 12 editions"
    assert candidate.content["description"] == "Fantasy, Adventure"
    assert candidate.content["text"] == "Allen & Unwin"
    assert candidate.content["url"] == "https://openlibrary.org/works/OL1W"
    assert candidate.content["image_url"] == "https://covers.openlibrary.org/b/id/12345-L.jpg"
    assert candidate.metadata["source"] == "open_library"


def test_rerank_book_search_result_reorders_results_and_preserves_metadata() -> None:
    llm = MockLLM([
        '{"ranked_ids": ["/works/OL7W", "/works/OL6W", "/works/OL5W", "/works/OL4W", "/works/OL3W", "/works/OL2W"]}'
    ])
    response = BookSearchResult.model_validate(
        {
            "numFound": 7,
            "docs": [
                {"key": "/works/OL1W", "title": "Book One"},
                {"key": "/works/OL2W", "title": "Book Two"},
                {"key": "/works/OL3W", "title": "Book Three"},
                {"key": "/works/OL4W", "title": "Book Four"},
                {"key": "/works/OL5W", "title": "Book Five"},
                {"key": "/works/OL6W", "title": "Book Six"},
                {"key": "/works/OL7W", "title": "Book Seven"},
            ],
        }
    )

    reranked = rerank_book_search_result(response, goal="best fantasy books", llm=llm)

    assert [book.key for book in reranked.docs] == [
        "/works/OL7W",
        "/works/OL6W",
        "/works/OL5W",
        "/works/OL4W",
        "/works/OL3W",
        "/works/OL2W",
    ]
    assert reranked.retrieved_count == 7
    assert reranked.reranked is True


def test_rerank_book_search_result_skips_llm_when_result_count_is_at_or_below_limit() -> None:
    llm = MockLLM([
        '{"ranked_ids": ["/works/OL2W", "/works/OL1W"]}'
    ])
    response = BookSearchResult.model_validate(
        {
            "numFound": 2,
            "docs": [
                {"key": "/works/OL1W", "title": "Book One"},
                {"key": "/works/OL2W", "title": "Book Two"},
            ],
        }
    )

    reranked = rerank_book_search_result(response, goal="best fantasy books", llm=llm)

    assert [book.key for book in reranked.docs] == [
        "/works/OL1W",
        "/works/OL2W",
    ]
    assert reranked.retrieved_count == 2
    assert reranked.reranked is True
    assert llm.last_prompt is None
    assert DEFAULT_BOOK_SEARCH_LIMIT == 20
