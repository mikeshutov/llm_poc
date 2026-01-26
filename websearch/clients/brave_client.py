from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests
from requests import Response

from models.search_type import SearchType


class BraveSearchError(RuntimeError):
    pass


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

        return resp.json()


    def web_search(
        self,
        q: str,
        search_type: SearchType,
        count: int = 10,
        offset: int = 0,
        country: str = "CA",
        search_lang: str = "en",
        **extra_params: Any,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "q": q,
            "count": count,
            "offset": offset,
            "country": country,
            "search_lang": search_lang,
            **extra_params,
        }
        return self._get("/web/search", params)

    def news_search(self, q: str, **params: Any) -> Dict[str, Any]:
        return self._get("/news/search", {"q": q, **params})

    def suggest(self, q: str, *, country: str = "CA", count: int = 5, **params: Any) -> Dict[str, Any]:
        return self._get("/suggest/search", {"q": q, "country": country, "count": count, **params})


    #next i also want to add some catalog searching functions to go along our product search