import re
from typing import Optional

from llm.clients.embeddings import embed_text
from products.candidate_mapper import rerank_product_results
from products.constants import DEFAULT_PRODUCT_SEARCH_CANDIDATE_LIMIT
from products.models.product_query import ProductQuery
from products.models.product_result import ProductResult
from products.models.product_search_results import ProductSearchResults
from products.models.product_source import ProductSource
from products.repository.product_repository import ProductRepository
from integrations.brave.client import BraveSearchClient, BraveSearchError
from integrations.brave.models import ShoppingSearchResult


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
) -> list[ProductResult]:
    results: list[ProductResult] = []
    for idx, item in enumerate(payload.results):
        if not item.title.strip():
            continue
        if not _is_high_confidence_product_detail_url(item.url):
            continue
        results.append(
            ProductResult(
                id=item.url or f"web-{idx}",
                name=item.title or "Unknown product",
                description=item.description,
                category=None,
                color=None,
                style=None,
                gender=None,
                season=None,
                year=None,
                price=_extract_price(item.description),
                url=item.url,
                image_url=item.image_url,
                score=None,
                source=ProductSource.WEB,
            )
        )
        if len(results) >= limit:
            break
    return results


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
) -> ProductSearchResults:
    repo = ProductRepository()
    query_embedding = embed_text(query_text or "")
    internal_results = repo.search_products(
        query_embedding=query_embedding,
        product_filters=product_filters,
        limit=DEFAULT_PRODUCT_SEARCH_CANDIDATE_LIMIT,
    )
    internal_results = rerank_product_results(internal_results, goal=query_text)
    return ProductSearchResults(internal_results=internal_results, external_results=[])


def find_products_web(
    query_text: str,
) -> ProductSearchResults:
    web_query = (query_text or "").strip() or "products"
    external_results: list[ProductResult] = []
    try:
        brave_client = BraveSearchClient()
        web_payload = brave_client.shopping_search(web_query, count=DEFAULT_PRODUCT_SEARCH_CANDIDATE_LIMIT)
        external_results = _web_results_to_products(web_payload, DEFAULT_PRODUCT_SEARCH_CANDIDATE_LIMIT)
    except (ValueError, BraveSearchError):
        pass
    external_results = rerank_product_results(external_results, goal=query_text)
    return ProductSearchResults(internal_results=[], external_results=external_results)
