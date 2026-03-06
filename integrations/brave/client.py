from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Dict, Optional

from integrations.models.web_search_params import WebSearchParams
from integrations.http_client import HttpClient, HttpClientError, DEFAULT_TTL
from integrations.brave.models import (
    NewsResult,
    NewsSearchResponse,
    ShoppingSearchResult,
    SuggestResponse,
    WebSearchResponse,
    WebSearchResult,
)
from integrations.brave.normalize import normalize_web_item


class BraveSearchError(RuntimeError):
    pass


@dataclass
class BraveSearchClient:
    api_key: str
    base_url: str = "https://api.search.brave.com/res/v1"
    timeout_s: float = 20.0
    max_retries: int = 3
    backoff_s: float = 0.75
    ttl: timedelta = field(default=DEFAULT_TTL)
    http: HttpClient = field(default=None)

    def __post_init__(self):
        if self.http is None:
            self.http = HttpClient(
                timeout_s=self.timeout_s,
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": self.api_key,
                },
                ttl=self.ttl,
            )

    @classmethod
    def from_env(cls, http: HttpClient | None = None, ttl: timedelta = DEFAULT_TTL) -> "BraveSearchClient":
        api_key = os.getenv("BRAVE_SEARCH_API_KEY")
        if not api_key:
            raise ValueError("Missing BRAVE_SEARCH_API_KEY in environment")
        return cls(api_key=api_key, http=http, ttl=ttl)

    def _get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        try:
            return self.http.get(url, params)
        except HttpClientError as e:
            raise BraveSearchError(str(e)) from e

    def web_search(self, params: WebSearchParams) -> WebSearchResponse:
        raw = self._get("/web/search", params.to_api_params())
        web_payload = raw.get("web") if isinstance(raw, dict) else None
        raw_results = web_payload.get("results", []) if isinstance(web_payload, dict) else []
        results = [normalize_web_item(item) for item in raw_results if isinstance(item, dict)]
        return WebSearchResponse(query=params.q, results=results)

    def news_search(self, q: str, **params: Any) -> NewsSearchResponse:
        raw = self._get("/news/search", {"q": q, **params})
        news_payload = raw.get("news") if isinstance(raw, dict) else None
        raw_results = news_payload.get("results", []) if isinstance(news_payload, dict) else []
        results = [
            NewsResult(
                title=str(item.get("title") or ""),
                url=item.get("url") if isinstance(item.get("url"), str) else None,
                description=str(item.get("description") or ""),
                age=item.get("age") if isinstance(item.get("age"), str) else None,
            )
            for item in raw_results
            if isinstance(item, dict)
        ]
        return NewsSearchResponse(query=q, results=results)

    def suggest(
        self,
        q: str,
        *,
        country: str = "CA",
        count: int = 5,
        **params: Any,
    ) -> SuggestResponse:
        raw = self._get("/suggest/search", {"q": q, "country": country, "count": count, **params})
        suggestions: list[str] = []
        if isinstance(raw, dict):
            for item in raw.get("results", []):
                if isinstance(item, dict):
                    s = item.get("query") or item.get("suggestion") or ""
                    if s:
                        suggestions.append(str(s))
        elif isinstance(raw, list):
            for item in raw:
                if isinstance(item, str):
                    suggestions.append(item)
                elif isinstance(item, dict):
                    s = item.get("query") or item.get("suggestion") or ""
                    if s:
                        suggestions.append(str(s))
        return SuggestResponse(query=q, suggestions=suggestions)

    def shopping_search(
        self,
        q: str,
        *,
        sources: Optional[list[str]] = None,
        count: int = 10,
        offset: int = 0,
        country: str = "US",
        search_lang: str = "en",
        **extra_params: Any,
    ) -> ShoppingSearchResult:
        if not q or not q.strip():
            raise ValueError("Query must be a non-empty string.")
        selected_source = "amazon"
        combined_results: list[WebSearchResult] = []

        for source in [selected_source]:
            source_key = source.strip().lower()
            if source_key == "amazon":
                query = f"{q} site:amazon.com"
            elif source_key in {"google_shopping", "google-shopping", "gshopping"}:
                query = f"{q} site:google.com/search"
            else:
                query = f"{q} site:{source}"

            resp = self.web_search(
                WebSearchParams(
                    q=query,
                    count=count,
                    offset=offset,
                    country=country,
                    search_lang=search_lang,
                    extra_params=extra_params,
                )
            )
            combined_results.extend(resp.results)

        return ShoppingSearchResult(
            query=q,
            sources=[selected_source],
            results=combined_results,
        )
