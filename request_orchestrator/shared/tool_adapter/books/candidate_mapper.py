from __future__ import annotations

from typing import Any

from common.utils import normalize_text, normalize_values, unique_normalized_values
from integrations.open_library.models import BookDoc, BookSearchResult
from request_orchestrator.shared.tool_adapter.books.constants import DEFAULT_BOOK_SEARCH_LIMIT
from reranker import RerankerCandidate, rerank_candidates


def _book_description(book: BookDoc) -> str | None:
    description = book.description
    if isinstance(description, dict):
        description = description.get("value")
    descriptions = normalize_values(description) or normalize_values(book.first_sentence)
    return " ".join(descriptions) if descriptions else None


def book_to_candidate(book: BookDoc) -> RerankerCandidate:
    author_names = unique_normalized_values(book.author_name)
    subjects = unique_normalized_values(book.subject, exclude_prefixes=("nyt:",))
    publishers = unique_normalized_values(book.publisher)
    publish_dates = normalize_values(book.publish_date)
    subtitle = normalize_text(book.subtitle) if book.subtitle else None

    return RerankerCandidate(
        id=book.key,
        fields={
            "title": normalize_text(book.title) or book.title,
            "subtitle": subtitle,
            "summary": _book_description(book),
            "authors": author_names,
            "subjects": subjects,
            "publishers": publishers,
            "languages": normalize_values(book.language),
            "first_publish_year": book.first_publish_year,
            "edition_count": book.edition_count,
            "number_of_pages": book.number_of_pages_median,
            "publish_dates": publish_dates,
        },
    )


def rerank_book_search_result(
    response: BookSearchResult,
    *,
    goal: str | None = None,
    llm: Any | None = None,
) -> BookSearchResult:
    retrieved_count = len(response.docs)
    if not response.docs:
        return BookSearchResult(
            numFound=response.num_found,
            start=response.start,
            docs=[],
            retrieved_count=0,
            reranked=True,
        )

    candidates = [book_to_candidate(book) for book in response.docs]
    ranked_candidates = rerank_candidates(
        candidates,
        goal=goal,
        llm=llm,
    )
    book_by_id = {book.key: book for book in response.docs}
    ranked_books: list[BookDoc] = []
    seen_ids: set[str] = set()

    for candidate in ranked_candidates:
        book = book_by_id.get(candidate.id)
        if book is None or book.key in seen_ids:
            continue
        ranked_books.append(book)
        seen_ids.add(book.key)

    return BookSearchResult(
        numFound=response.num_found,
        start=response.start,
        docs=ranked_books[:DEFAULT_BOOK_SEARCH_LIMIT],
        retrieved_count=retrieved_count,
        reranked=True,
    )
