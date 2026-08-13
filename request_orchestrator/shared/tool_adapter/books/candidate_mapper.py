from __future__ import annotations

from typing import Any

from common.utils import normalize_text
from integrations.open_library.models import BookDoc, BookSearchResult
from request_orchestrator.shared.tool_adapter.books.constants import DEFAULT_BOOK_SEARCH_LIMIT
from reranker import Candidate, rerank_candidates


def book_to_candidate(book: BookDoc) -> Candidate:
    author_names = [cleaned for author in (book.author_name or []) if (cleaned := normalize_text(author))]
    subjects = [cleaned for subject in (book.subject or []) if (cleaned := normalize_text(subject))]
    publishers = [cleaned for publisher in (book.publisher or []) if (cleaned := normalize_text(publisher))]

    summary_parts = [
        cleaned
        for cleaned in (
            ", ".join(author_names) if author_names else None,
            str(book.first_publish_year) if book.first_publish_year is not None else None,
            f"{book.edition_count} editions" if book.edition_count is not None else None,
        )
        if cleaned is not None
    ]

    return Candidate(
        id=book.key,
        title=normalize_text(book.title) or book.title,
        content={
            "name": normalize_text(book.title),
            "summary": ". ".join(summary_parts) if summary_parts else None,
            "description": ", ".join(subjects) if subjects else None,
            "text": ", ".join(publishers) if publishers else None,
            "url": f"https://openlibrary.org{book.key}" if book.key else None,
            "image_url": f"https://covers.openlibrary.org/b/id/{book.cover_i}-L.jpg" if book.cover_i is not None else None,
        },
        attributes={
            "authors": author_names,
            "subjects": subjects,
            "languages": list(book.language or []),
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
