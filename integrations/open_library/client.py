from __future__ import annotations

from datetime import timedelta
from typing import Any

from common.http import HttpClient, HttpClientError, DEFAULT_TTL
from integrations.open_library.constants import DEFAULT_BOOK_SEARCH_LIMIT
from integrations.open_library.models import BookSearchResult

OPEN_LIBRARY_BASE_URL = "https://openlibrary.org"
OPEN_LIBRARY_WORK_URL_TEMPLATE = "https://openlibrary.org{work_key}"
OPEN_LIBRARY_COVER_IMAGE_URL_TEMPLATE = "https://covers.openlibrary.org/b/id/{cover_id}-L.jpg"
OPEN_LIBRARY_BOOK_FIELDS = ",".join(
    (
        "key",
        "title",
        "subtitle",
        "description",
        "first_sentence",
        "author_name",
        "first_publish_year",
        "edition_count",
        "number_of_pages_median",
        "publish_date",
        "subject",
        "publisher",
        "language",
        "cover_i",
    )
)


class OpenLibraryClientError(RuntimeError):
    pass


class OpenLibraryClient:
    def __init__(
        self,
        base_url: str = OPEN_LIBRARY_BASE_URL,
        timeout_s: float = 20.0,
        ttl: timedelta = DEFAULT_TTL,
    ):
        self.base_url = base_url.rstrip("/")
        self._http = HttpClient(
            timeout_s=timeout_s,
            ttl=ttl,
        )

    def search(
        self,
        query: str,
        limit: int = DEFAULT_BOOK_SEARCH_LIMIT,
        page: int = 1,
    ) -> BookSearchResult:
        q = (query or "").strip()
        if not q:
            raise ValueError("Search query must not be empty.")

        url = f"{self.base_url}/search.json"
        params: dict[str, Any] = {
            "q": q,
            "limit": limit,
            "page": page,
            "fields": OPEN_LIBRARY_BOOK_FIELDS,
        }
        try:
            payload = self._http.get(url, params)
        except HttpClientError as e:
            raise OpenLibraryClientError(str(e)) from e
        if not isinstance(payload, dict):
            raise OpenLibraryClientError("Unexpected response from Open Library API.")
        return BookSearchResult.model_validate(payload)
