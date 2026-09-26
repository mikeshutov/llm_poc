from __future__ import annotations

from typing import Any

from common.utils import normalize_text
from integrations.open_library import OPEN_LIBRARY_COVER_IMAGE_URL_TEMPLATE, OPEN_LIBRARY_WORK_URL_TEMPLATE
from integrations.open_library.models import BookDoc, BookSearchResult
from request_orchestrator.shared.tool_adapter.books.constants import DEFAULT_BOOK_SEARCH_LIMIT
from reranker import Candidate, rerank_candidates


def _normalized_values(values: object) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, (list, tuple)):
        values = [values]
    return [cleaned for value in values if (cleaned := normalize_text(str(value)))]


def _unique_values(values: object, *, exclude_prefixes: tuple[str, ...] = ()) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in _normalized_values(values):
        if any(value.casefold().startswith(prefix.casefold()) for prefix in exclude_prefixes):
            continue
        normalized = value.casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(value)
    return unique


def _book_description(book: BookDoc) -> str | None:
    description = book.description
    if isinstance(description, dict):
        description = description.get("value")
    descriptions = _normalized_values(description) or _normalized_values(book.first_sentence)
    return " ".join(descriptions) if descriptions else None


def book_to_candidate(book: BookDoc) -> Candidate:
    author_names = _unique_values(book.author_name)
    subjects = _unique_values(book.subject, exclude_prefixes=("nyt:",))
    publishers = _unique_values(book.publisher)
    publish_dates = _normalized_values(book.publish_date)
    subtitle = normalize_text(book.subtitle) if book.subtitle else None

    return Candidate(
        id=book.key,
        title=normalize_text(book.title) or book.title,
        content={
            "name": normalize_text(book.title),
            "summary": _book_description(book),
            "url": OPEN_LIBRARY_WORK_URL_TEMPLATE.format(work_key=book.key) if book.key else None,
            "image_url": OPEN_LIBRARY_COVER_IMAGE_URL_TEMPLATE.format(cover_id=book.cover_i) if book.cover_i is not None else None,
        },
        attributes={
            "authors": author_names,
            "subjects": subjects,
            "publishers": publishers,
            "languages": _normalized_values(book.language),
            "subtitle": subtitle,
            "publish_dates": publish_dates,
            "first_publish_year": book.first_publish_year,
            "edition_count": book.edition_count,
            "number_of_pages": book.number_of_pages_median,
        },
        metadata={
            "source": "open_library",
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
