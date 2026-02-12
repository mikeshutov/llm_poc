import re
from typing import Optional
from products.models.product_result import ProductResult
from products.models.product_search_results import ProductSearchResults
from products.repository.product_repository import ProductRepository
from websearch.models.common_properties import CommonProperties
from products.models.product_source import ProductSource
from websearch.clients.brave_client import BraveSearchClient, BraveSearchError, ShoppingSearchResult
from websearch.clients.brave_normalize import WebSearchItem, normalize_web_item
from websearch.query_builder import build_web_query


def _extract_price(text: str) -> Optional[float]:
    if not text:
        return None
    match = re.search(r"\$([0-9]+(?:\.[0-9]{2})?)", text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _web_results_to_products(payload: ShoppingSearchResult, limit: int) -> list[ProductResult]:
    results: list[ProductResult] = []
    for idx, item in enumerate(payload.results):
        normalized = normalize_web_item(item)
        if not _is_product_detail_url(normalized.url):
            continue
        price = _extract_price(normalized.description)
        results.append(
            ProductResult(
                id=normalized.url or f"web-{idx}",
                name=normalized.title or normalized.description or "Unknown product",
                category=None,
                color=None,
                style=None,
                gender=None,
                season=None,
                year=None,
                price=price,
                url=normalized.url,
                image_url=normalized.image_url,
                score=None,
                source=ProductSource.WEB,
            )
        )
        if len(results) >= limit:
            break
    return results


def _is_product_detail_url(url: Optional[str]) -> bool:
    if not url:
        return False
    u = url.lower()
    if "amazon." in u:
        return "/dp/" in u or "/gp/product/" in u
    if "shopping.google." in u:
        return "/product/" in u
    return True


def find_products(query_text: str, common_filters: Optional[CommonProperties], web_count: int = 5) -> ProductSearchResults:
    repo = ProductRepository()
    internal_results = repo.search_products_filtered(
        query_text=query_text,
        common_filters=common_filters,
        product_filters=None, # we will utilize this later on.
        limit=10,
    )

    web_query = build_web_query(common_filters).strip()
    external_results: list[ProductResult] = []
    if web_query:
        try:
            brave_client = BraveSearchClient.from_env()
            web_payload = brave_client.shopping_search(web_query, count=20)
            print(web_payload)
            external_results = _web_results_to_products(web_payload, max(1, web_count))
        except (ValueError, BraveSearchError):
            pass

    return ProductSearchResults(
        internal_results=internal_results,
        external_results=external_results,
    )
