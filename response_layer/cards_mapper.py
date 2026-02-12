from typing import Any, Iterable, List

from products.models.product_result import ProductResult
from products.models.product_search_results import ProductSearchResults


def product_results_to_cards(results: ProductSearchResults, limit: int = 10) -> List[dict]:
    cards: list[dict] = []
    for item in [*results.internal_results, *results.external_results]:
        cards.append(
            {
                "id": item.id,
                "name": item.name,
                "description": None,
                "price": item.price,
                "url": item.url,
                "image_url": item.image_url,
                "source": item.source.value if hasattr(item.source, "value") else str(item.source),
            }
        )
        if len(cards) >= limit:
            break
    return cards


# to fix this is just reusing the same properties
def news_results_to_cards(payload: dict, limit: int = 5) -> list[dict]:
    results = payload.get("news", {}).get("results", [])
    cards: list[dict] = []
    for idx, item in enumerate(results):
        title = item.get("title") or item.get("name") or ""
        url = item.get("url") or item.get("link")
        description = item.get("description") or item.get("snippet") or ""
        thumbnail = item.get("thumbnail")
        image_url = None
        if isinstance(thumbnail, dict):
            image_url = thumbnail.get("original") or thumbnail.get("src")
        elif isinstance(thumbnail, str):
            image_url = thumbnail

        cards.append(
            {
                "id": url or f"news-{idx}",
                "name": title or "Untitled article",
                "description": description,
                "price": None,
                "url": url,
                "image_url": image_url,
                "source": "news",
            }
        )
        if len(cards) >= limit:
            break
    return cards
