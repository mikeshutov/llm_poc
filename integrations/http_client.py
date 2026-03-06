from __future__ import annotations

from datetime import timedelta
from typing import Any

import requests

from cache.rest_cache_repository import RestCacheRepository

DEFAULT_TTL = timedelta(hours=1)


class HttpClientError(RuntimeError):
    pass


class HttpClient:
    def __init__(
        self,
        timeout_s: float = 20.0,
        headers: dict[str, str] | None = None,
        cache: RestCacheRepository | None = None,
        ttl: timedelta = DEFAULT_TTL,
    ):
        self._timeout_s = timeout_s
        self._headers = headers or {}
        self._cache = cache
        self._ttl = ttl

    def get(self, url: str, params: dict[str, Any] | None = None) -> Any:
        params = params or {}
        if self._cache:
            cached = self._cache.get(url, params)
            if cached is not None:
                return cached

        resp = requests.get(url, params=params, headers=self._headers, timeout=self._timeout_s)
        if not resp.ok:
            raise HttpClientError(f"HTTP {resp.status_code} on {url}: {resp.text[:500]}")

        payload = resp.json()

        if self._cache:
            self._cache.put(url, params, payload, self._ttl)

        return payload
