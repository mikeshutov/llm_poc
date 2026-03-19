from __future__ import annotations

from datetime import timedelta
from typing import Any

from common.http import HttpClient, HttpClientError, DEFAULT_TTL
from integrations.quotable.models import Quote


class QuotableClientError(RuntimeError):
    pass


class QuotableClient:
    def __init__(
        self,
        base_url: str = "https://api.quotable.kurokeita.dev/api",
        timeout_s: float = 10.0,
        ttl: timedelta = DEFAULT_TTL,
    ):
        self.base_url = base_url.rstrip("/")
        self._http = HttpClient(
            timeout_s=timeout_s,
            ttl=ttl,
        )

    def random(self) -> Quote:
        url = f"{self.base_url}/quotes/random"
        try:
            payload = self._http.get(url)
        except HttpClientError as e:
            raise QuotableClientError(str(e)) from e
        if not isinstance(payload, dict):
            raise QuotableClientError("Unexpected response from Quotable API.")
        data = payload.get("data")
        if not isinstance(data, list) or not data:
            raise QuotableClientError("No quote data returned from Quotable API.")
        return self._parse_quote(data[0])

    def search(self, query: str) -> list[Quote]:
        query = (query or "").strip()
        if not query:
            raise ValueError("Search query must not be empty.")
        url = f"{self.base_url}/quotes"
        try:
            payload = self._http.get(url, {"query": query})
        except HttpClientError as e:
            raise QuotableClientError(str(e)) from e
        if not isinstance(payload, dict):
            raise QuotableClientError("Unexpected response from Quotable API.")
        data = payload.get("data")
        if not isinstance(data, list):
            return []
        return [self._parse_quote(item) for item in data if isinstance(item, dict)]

    def _parse_quote(self, data: dict[str, Any]) -> Quote:
        author_obj = data.get("author") or {}
        author_name = author_obj.get("name", "") if isinstance(author_obj, dict) else str(author_obj)
        tags = data.get("tags") or []
        return Quote(
            content=data.get("content", ""),
            author=author_name,
            tags=[t for t in tags if isinstance(t, str)],
        )
