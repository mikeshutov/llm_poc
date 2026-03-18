from __future__ import annotations

from datetime import timedelta
from typing import Any

from common.http import HttpClient, HttpClientError, DEFAULT_TTL
from integrations.open_library.models import BookSearchResult


class OpenLibraryClientError(RuntimeError):
    pass


class OpenLibraryClient:
    def __init__(
        self,
        base_url: str = "https://openlibrary.org",
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
        limit: int = 10,
        page: int = 1,
    ) -> BookSearchResult:
        q = (query or "").strip()
        if not q:
            raise ValueError("Search query must not be empty.")

        url = f"{self.base_url}/search.json"
        params: dict[str, Any] = {"q": q, "limit": limit, "page": page}
        try:
            payload = self._http.get(url, params)
        except HttpClientError as e:
            raise OpenLibraryClientError(str(e)) from e
        if not isinstance(payload, dict):
            raise OpenLibraryClientError("Unexpected response from Open Library API.")
        return BookSearchResult.model_validate(payload)
