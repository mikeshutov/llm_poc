from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
import re

import requests


class WebFetchError(RuntimeError):
    """Raised when a web page cannot be fetched or parsed."""


@dataclass(frozen=True)
class FetchedPage:
    url: str
    title: str
    description: str
    text: str
    status_code: int
    content_type: str


class _PageParser(HTMLParser):
    _IGNORED_TAGS = {"script", "style", "noscript", "template", "svg"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.description = ""
        self._ignored_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in self._IGNORED_TAGS:
            self._ignored_depth += 1
            return
        if tag == "title":
            self._in_title = True
        if tag == "meta":
            attributes = dict(attrs)
            if attributes.get("name", "").lower() == "description":
                self.description = (attributes.get("content") or "").strip()
        if tag in {"br", "p", "div", "li", "article", "section", "h1", "h2", "h3"}:
            self.text_parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self._IGNORED_TAGS:
            self._ignored_depth = max(0, self._ignored_depth - 1)
            return
        if tag == "title":
            self._in_title = False
        if tag in {"p", "div", "li", "article", "section", "h1", "h2", "h3"}:
            self.text_parts.append(" ")

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        cleaned = data.strip()
        if not cleaned:
            return
        if self._in_title:
            self.title_parts.append(cleaned)
        self.text_parts.append(cleaned)


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


@dataclass
class WebFetchClient:
    timeout_s: float = 20.0
    max_text_chars: int = 30_000
    user_agent: str = "POCProductSearch/1.0"

    def fetch(self, url: str) -> FetchedPage:
        normalized_url = url.strip()
        if not normalized_url:
            raise ValueError("url is required and cannot be blank.")
        if not normalized_url.startswith(("http://", "https://")):
            raise ValueError("url must start with http:// or https://.")
        try:
            response = requests.get(
                normalized_url,
                headers={"User-Agent": self.user_agent, "Accept": "text/html,application/xhtml+xml"},
                timeout=self.timeout_s,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise WebFetchError(f"Unable to fetch {normalized_url}: {exc}") from exc

        content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type and content_type not in {"text/html", "application/xhtml+xml"}:
            text = response.text[: self.max_text_chars].strip()
            return FetchedPage(response.url or normalized_url, "", "", text, response.status_code, content_type)

        parser = _PageParser()
        parser.feed(response.text)
        parser.close()
        return FetchedPage(
            response.url or normalized_url,
            _normalize_text(" ".join(parser.title_parts)),
            _normalize_text(parser.description),
            _normalize_text(" ".join(parser.text_parts))[: self.max_text_chars],
            response.status_code,
            content_type or "text/html",
        )
