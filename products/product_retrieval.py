import re
from typing import Optional
from urllib.parse import urlparse

from llm.clients.embeddings import embed_text
from products.models.product_query import ProductQuery
from products.models.product_result import ProductResult
from products.models.product_search_results import ProductSearchResults
from products.models.product_source import ProductSource
from products.repository.product_repository import ProductRepository
from integrations.brave.client import BraveSearchClient, BraveSearchError
from integrations.brave.models import ShoppingSearchResult
from integrations.brave.query_builder import build_web_query


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


def _web_results_to_products(
    payload: ShoppingSearchResult,
    limit: int,
) -> tuple[list[ProductResult], dict[str, object]]:
    results: list[ProductResult] = []
    fallback_domain_breakdown: dict[str, int] = {}
    for idx, item in enumerate(payload.results):
        if not item.title.strip():
            continue
        if not _is_high_confidence_product_detail_url(item.url):
            continue
        price = _extract_price(item.description)
        if item.url:
            domain = (urlparse(item.url).hostname or "unknown").lower()
            fallback_domain_breakdown[domain] = fallback_domain_breakdown.get(domain, 0) + 1
        results.append(
            ProductResult(
                id=item.url or f"web-{idx}",
                name=item.title or item.description or "Unknown product",
                category=None,
                color=None,
                style=None,
                gender=None,
                season=None,
                year=None,
                price=price,
                url=item.url,
                image_url=item.image_url,
                score=None,
                source=ProductSource.WEB,
            )
        )
        if len(results) >= limit:
            break
    return results, {
        "fallback_candidate_count": len(payload.results),
        "fallback_valid_count": len(results),
        "fallback_domain_breakdown": fallback_domain_breakdown,
    }


def _is_high_confidence_product_detail_url(url: Optional[str]) -> bool:
    if not url:
        return False
    u = url.lower()
    if "amazon." in u:
        return "/dp/" in u or "/gp/product/" in u
    if "walmart." in u:
        return "/ip/" in u
    if "bestbuy." in u:
        return "/site/" in u and bool(re.search(r"/\d+\.p($|[/?])", u))
    if "target." in u:
        return "/p/" in u
    return False


def find_products(
    query_text: str,
    product_filters: Optional[ProductQuery] = None,
    web_count: int = 5,
    allow_web_fallback: bool = True,
) -> ProductSearchResults:
    repo = ProductRepository()
    query_embedding = embed_text(query_text or "")
    internal_results = repo.search_products(
        query_embedding=query_embedding,
        product_filters=product_filters,
        limit=10,
    )

    external_results: list[ProductResult] = []
    if not internal_results and allow_web_fallback:
        web_query = ((query_text or "").strip() or build_web_query(product_filters).strip() or "products")
        try:
            brave_client = BraveSearchClient.from_env()
            web_payload = brave_client.shopping_search(web_query, count=20)
            external_results, _ = _web_results_to_products(web_payload, max(1, web_count))
        except (ValueError, BraveSearchError):
            pass

    return ProductSearchResults(
        internal_results=internal_results,
        external_results=external_results,
    )
