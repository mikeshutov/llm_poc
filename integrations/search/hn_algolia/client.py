from __future__ import annotations

from datetime import timedelta
from typing import Any, Literal

from integrations.http_client import HttpClient, HttpClientError, DEFAULT_TTL
from integrations.search.hn_algolia.models import HnSearchResult


class HnAlgoliaClientError(RuntimeError):
    pass


HnSortBy = Literal["relevance", "date"]


class HnAlgoliaClient:
    def __init__(
        self,
        base_url: str = "https://hn.algolia.com/api/v1",
        timeout_s: float = 20.0,
        user_agent: str = "POCProductSearch/1.0 (HN Algolia client)",
        ttl: timedelta = DEFAULT_TTL,
        http: HttpClient | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self._http = http or HttpClient(
            timeout_s=timeout_s,
            headers={"Accept": "application/json", "User-Agent": user_agent},
            ttl=ttl,
        )

    def search(
        self,
        query: str,
        sort_by: HnSortBy = "relevance",
        tags: str | None = None,
        hits_per_page: int = 10,
        page: int = 0,
    ) -> HnSearchResult:
        q = (query or "").strip()
        if not q:
            raise ValueError("Search query must not be empty.")

        endpoint = "search_by_date" if sort_by == "date" else "search"
        url = f"{self.base_url}/{endpoint}"
        params: dict[str, Any] = {"query": q, "hitsPerPage": hits_per_page, "page": page}
        if tags:
            params["tags"] = tags

        try:
            payload = self._http.get(url, params)
        except HttpClientError as e:
            raise HnAlgoliaClientError(str(e)) from e
        if not isinstance(payload, dict):
            raise HnAlgoliaClientError("Unexpected response from HN Algolia API.")
        return HnSearchResult.model_validate(payload)
