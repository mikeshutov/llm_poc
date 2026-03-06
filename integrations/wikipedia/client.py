from __future__ import annotations

from datetime import timedelta
from typing import Any
from urllib.parse import quote

from integrations.http_client import HttpClient, HttpClientError, DEFAULT_TTL
from integrations.wikipedia.models import WikipediaPageSummary, WikipediaSearchResult


class WikipediaClientError(RuntimeError):
    pass


class WikipediaNotFoundError(WikipediaClientError):
    pass


class WikipediaClient:
    def __init__(
        self,
        base_url: str = "https://en.wikipedia.org",
        timeout_s: float = 20.0,
        user_agent: str = "POCProductSearch/1.0 (Wikipedia client)",
        ttl: timedelta = DEFAULT_TTL,
        http: HttpClient | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self._http = http or HttpClient(
            timeout_s=timeout_s,
            headers={"Accept": "application/json", "User-Agent": user_agent},
            ttl=ttl,
        )

    @property
    def api_url(self) -> str:
        return f"{self.base_url}/w/api.php"

    def _get_json(self, params: dict[str, Any]) -> dict[str, Any] | list[Any]:
        try:
            payload = self._http.get(self.api_url, params)
        except HttpClientError as e:
            raise WikipediaClientError(str(e)) from e
        if not isinstance(payload, (dict, list)):
            raise WikipediaClientError("Unexpected non-JSON response from Wikipedia API.")
        return payload

    def search(self, query: str, limit: int = 5) -> list[WikipediaSearchResult]:
        query_norm = (query or "").strip()
        if not query_norm:
            raise ValueError("Query must be a non-empty string.")
        if limit < 1:
            raise ValueError("Limit must be at least 1.")

        payload = self._get_json(
            {
                "action": "opensearch",
                "search": query_norm,
                "limit": limit,
                "namespace": 0,
                "format": "json",
            }
        )
        if not isinstance(payload, list) or len(payload) < 4:
            raise WikipediaClientError("Malformed Wikipedia OpenSearch response.")

        titles = payload[1] if isinstance(payload[1], list) else []
        descriptions = payload[2] if isinstance(payload[2], list) else []
        urls = payload[3] if isinstance(payload[3], list) else []

        results: list[WikipediaSearchResult] = []
        for title, description, url in zip(titles, descriptions, urls):
            if not isinstance(title, str) or not isinstance(url, str):
                continue
            results.append(
                WikipediaSearchResult(
                    title=title,
                    description=description if isinstance(description, str) else "",
                    url=url,
                )
            )
        return results

    def get_page_summary(self, title: str, sentences: int = 2) -> WikipediaPageSummary:
        title_norm = (title or "").strip()
        if not title_norm:
            raise ValueError("Title must be a non-empty string.")
        if sentences < 1:
            raise ValueError("Sentences must be at least 1.")

        payload = self._get_json(
            {
                "action": "query",
                "prop": "extracts|info",
                "exintro": 1,
                "explaintext": 1,
                "exsentences": sentences,
                "inprop": "url",
                "redirects": 1,
                "titles": title_norm,
                "format": "json",
            }
        )
        if not isinstance(payload, dict):
            raise WikipediaClientError("Malformed Wikipedia summary response.")

        query_payload = payload.get("query")
        pages = query_payload.get("pages") if isinstance(query_payload, dict) else None
        if not isinstance(pages, dict) or not pages:
            raise WikipediaClientError("Malformed Wikipedia summary response: missing pages.")

        first_page = next(iter(pages.values()))
        if not isinstance(first_page, dict):
            raise WikipediaClientError("Malformed Wikipedia summary response: page is not an object.")
        if "missing" in first_page:
            raise WikipediaNotFoundError(f"No page found for title '{title_norm}'.")

        resolved_title = str(first_page.get("title") or title_norm)
        summary = str(first_page.get("extract") or "").strip()
        page_id = first_page.get("pageid")
        page_id_value = int(page_id) if isinstance(page_id, int) else None
        url = first_page.get("canonicalurl") or first_page.get("fullurl")
        if not isinstance(url, str) or not url.strip():
            page_slug = quote(resolved_title.replace(" ", "_"), safe="()")
            url = f"{self.base_url}/wiki/{page_slug}"

        return WikipediaPageSummary(
            title=resolved_title,
            summary=summary,
            url=url,
            page_id=page_id_value,
        )
