from integrations.brave.models import (
    NewsResult,
    NewsSearchResponse,
    ShoppingSearchResult,
    SuggestResponse,
    WebSearchResponse,
    WebSearchResult,
)


def __getattr__(name: str):
    if name in {"BraveSearchClient", "BraveSearchError"}:
        from integrations.brave.client import BraveSearchClient, BraveSearchError

        return {
            "BraveSearchClient": BraveSearchClient,
            "BraveSearchError": BraveSearchError,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BraveSearchClient",
    "BraveSearchError",
    "NewsResult",
    "NewsSearchResponse",
    "ShoppingSearchResult",
    "SuggestResponse",
    "WebSearchResponse",
    "WebSearchResult",
]
