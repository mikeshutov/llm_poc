from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from requests.exceptions import RequestException

from integrations.open_library import OpenLibraryClient, BookSearchResult
from request_orchestrator.shared.tool_adapter.books.candidate_mapper import rerank_book_search_result
from request_orchestrator.shared.tool_adapter.books.constants import DEFAULT_BOOK_SEARCH_LIMIT

_open_library_client = OpenLibraryClient()


class SearchBooksArgs(BaseModel):
    query: str = Field(
        ...,
        description="Search query for books. Can be a title, author, subject, or general keyword.",
    )


@tool(
    "search_books",
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
def search_books(query: str) -> BookSearchResult | str:
    try:
        response = _open_library_client.search(query, limit=DEFAULT_BOOK_SEARCH_LIMIT)
        return rerank_book_search_result(response, goal=query)
    except RequestException as e:
        return f"Open Library service unavailable: {e}"
