from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

import requests
from requests import Response

from debug.rendering import debug_render_message
from websearch.models.web_search_params import WebSearchParams

class BraveSearchError(RuntimeError):
    pass


@dataclass(frozen=True)
class ShoppingSearchResult:
    query: str
    sources: List[str]
    results: List[Dict[str, Any]]
    raw: Dict[str, Any]


@dataclass
class BraveSearchClient:
    api_key: str
    base_url: str = "https://api.search.brave.com/res/v1"
    timeout_s: float = 20.0
    max_retries: int = 3
    backoff_s: float = 0.75

    @classmethod
    def from_env(cls) -> "BraveSearchClient":
        api_key = os.getenv("BRAVE_SEARCH_API_KEY")
        if not api_key:
            raise ValueError(f"Missing BRAVE_SEARCH_API_KEY in environment")
        return cls(api_key=api_key)

    def _headers(self) -> Dict[str, str]:
        return {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key,
        }

    def _get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"

        resp: Response = requests.get(
            url, headers=self._headers(), params=params, timeout=self.timeout_s
        )

        if not resp.ok:
            body = resp.text[:500]
            raise BraveSearchError(
                f"Brave API error {resp.status_code} on {path}: {body}"
            )

        payload = resp.json()
        debug_render_message(
            {
                "service": "brave_search",
                "path": path,
                "params": params,
                "status": resp.status_code,
                "response": payload,
            },
            "Debug: Brave search request",
        )
        return payload


    def web_search(
        self,
        params: WebSearchParams,
    ) -> Dict[str, Any]:
        return self._get("/web/search", params.to_api_params())

    def news_search(self, q: str, **params: Any) -> Dict[str, Any]:
        return self._get("/news/search", {"q": q, **params})

    def suggest(self, q: str, *, country: str = "CA", count: int = 5, **params: Any) -> Dict[str, Any]:
        return self._get("/suggest/search", {"q": q, "country": country, "count": count, **params})


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
        responses: Dict[str, Any] = {}
        combined_results: list[Dict[str, Any]] = []

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
            responses[source_key] = resp

            web_results = resp.get("web", {}).get("results", [])
            if isinstance(web_results, list):
                combined_results.extend(web_results)

        return ShoppingSearchResult(
            query=q,
            sources=[selected_source],
            results=combined_results,
            raw=responses,
        )
