from __future__ import annotations

from integrations.search.brave.models import WebSearchResult


def normalize_web_item(item: dict) -> WebSearchResult:
    return WebSearchResult.model_validate(item)
