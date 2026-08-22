from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from requests.exceptions import RequestException

from integrations.open_library import (
    OPEN_LIBRARY_COVER_IMAGE_URL_TEMPLATE,
    OPEN_LIBRARY_WORK_URL_TEMPLATE,
    OpenLibraryClient,
)
from integrations.open_library.models import BookDoc, BookSearchResult
from request_orchestrator.models.evidence import EvidenceUrl, EvidenceUrlType, EvidenceView, HydratedEvidence, ToolResult
from request_orchestrator.shared.tool_adapter.books.candidate_mapper import rerank_book_search_result
from request_orchestrator.shared.tool_adapter.books.constants import DEFAULT_BOOK_SEARCH_LIMIT
from tool.constants import TOOL_NAME_SEARCH_BOOKS
from tool.constants import TOOL_RESULT_TYPE_BOOK_RESULTS

_open_library_client = OpenLibraryClient()


class SearchBooksArgs(BaseModel):
    query: str = Field(
        ...,
        description="Search query for books. Can be a title, author, subject, or general keyword.",
    )


class BookSearchMetadata(BaseModel):
    authors: list[str] = []
    subjects: list[str] = []
    languages: list[str] = []


def _book_summary(book: BookDoc) -> str:
    parts: list[str] = []
    if book.author_name:
        parts.append(", ".join(book.author_name))
    if book.first_publish_year is not None:
        parts.append(str(book.first_publish_year))
    if book.edition_count is not None:
        parts.append(f"{book.edition_count} editions")
    return ". ".join(parts) if parts else f"Book result for {book.title}."


def _tool_result(result: BookSearchResult) -> ToolResult:
    hydrated_evidence: list[HydratedEvidence] = []
    evidence_views: list[EvidenceView] = []
    for book in result.docs:
        url = OPEN_LIBRARY_WORK_URL_TEMPLATE.format(work_key=book.key).strip() if book.key else ""
        image_url = OPEN_LIBRARY_COVER_IMAGE_URL_TEMPLATE.format(cover_id=book.cover_i).strip() if book.cover_i is not None else ""
        metadata = BookSearchMetadata(
            authors=list(book.author_name or []),
            subjects=list(book.subject or []),
            languages=list(book.language or []),
        )
        hydrated = HydratedEvidence(
            item_id=book.key,
            tool_name=TOOL_NAME_SEARCH_BOOKS,
            title=book.title,
            summary=_book_summary(book),
            urls=[EvidenceUrl(url=url, url_type=EvidenceUrlType.WEBSITE)] if url else [],
            image_url=image_url,
            source=TOOL_NAME_SEARCH_BOOKS,
            entity_type=TOOL_RESULT_TYPE_BOOK_RESULTS,
            metadata=metadata.model_dump(exclude_none=True),
            raw_payload=book,
        )
        hydrated_evidence.append(hydrated)
        evidence_views.append(
            EvidenceView(
                item_id=hydrated.item_id,
                title=hydrated.title,
                summary=hydrated.summary,
                metadata=dict(hydrated.metadata),
            )
        )
    return ToolResult(result=result, evidence_views=evidence_views, hydrated_evidence=hydrated_evidence)




@tool(
    TOOL_NAME_SEARCH_BOOKS,
    args_schema=SearchBooksArgs,
    description="""
Search the Open Library catalog for books by a title, author, subject, or keyword.

Required fields:
- query (string)

Returns a list of books with title, authors, first publish year, edition count, and subjects.

Example valid call:
{
  "query": "tolkien lord of the rings"
}
""",
)
def search_books(query: str) -> ToolResult:
    try:
        response = _open_library_client.search(query, limit=DEFAULT_BOOK_SEARCH_LIMIT)
        return _tool_result(rerank_book_search_result(response, goal=query))
    except RequestException as e:
        return ToolResult.error(f"Open Library service unavailable: {e}")
