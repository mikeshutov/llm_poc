from typing import Any, Iterable, List

from integrations.brave.models import NewsSearchResponse
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


def news_response_to_cards(response: NewsSearchResponse, limit: int = 5) -> list[dict]:
    cards: list[dict] = []
    for idx, item in enumerate(response.results):
        cards.append(
            {
                "id": item.url or f"news-{idx}",
                "name": item.title or "Untitled article",
                "description": item.description,
                "price": None,
                "url": item.url,
                "image_url": item.thumbnail_url,
                "source": "news",
            }
        )
        if len(cards) >= limit:
            break
    return cards
