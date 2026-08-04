from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from requests.exceptions import RequestException

from integrations.open_library import OpenLibraryClient, BookSearchResult

_open_library_client = OpenLibraryClient()


class SearchBooksArgs(BaseModel):
    query: str = Field(
        ...,
        description="Search query for books. Can be a title, author, subject, or general keyword.",
    )
    limit: Optional[int] = Field(
        default=10,
        description="Maximum number of results to return (1–100). Defaults to 10.",
    )


@tool(
    "search_books",
    args_schema=SearchBooksArgs,
    description="""
Search the Open Library catalog for books by a title, author, subject, or keyword.

Required fields:
- query (string)

Optional fields:
- limit (integer, default 10)

Returns a list of books with title, authors, first publish year, edition count, and subjects.

Example valid call:
{
  "query": "tolkien lord of the rings"
}
""",
)
def search_books(query: str, limit: int = 10) -> BookSearchResult | str:
    try:
        return _open_library_client.search(query, limit=limit)
    except RequestException as e:
        return f"Open Library service unavailable: {e}"
