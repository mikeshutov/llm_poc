import re
from urllib.parse import urlparse
from typing import Optional

from intent_processing.product_embeddings import embed_text
from products.models.product_query import ProductQuery
from products.models.product_result import ProductResult
from products.models.product_search_results import ProductSearchResults
from products.models.product_source import ProductSource
from products.repository.product_repository import ProductRepository
from websearch.clients.brave_client import BraveSearchClient, BraveSearchError, ShoppingSearchResult
from websearch.clients.brave_normalize import normalize_web_item
from websearch.models.common_properties import CommonProperties
from websearch.query_builder import build_web_query

_LAST_FALLBACK_META: dict[str, object] = {
    "fallback_candidate_count": 0,
    "fallback_valid_count": 0,
    "fallback_domain_breakdown": {},
}


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


def _web_results_to_products(payload: ShoppingSearchResult, limit: int) -> tuple[list[ProductResult], dict[str, object]]:
    results: list[ProductResult] = []
    fallback_domain_breakdown: dict[str, int] = {}
    for idx, item in enumerate(payload.results):
        normalized = normalize_web_item(item)
        if not normalized.title.strip():
            continue
        if not _is_high_confidence_product_detail_url(normalized.url):
            continue
        price = _extract_price(normalized.description)
        if normalized.url:
            domain = (urlparse(normalized.url).hostname or "unknown").lower()
            fallback_domain_breakdown[domain] = fallback_domain_breakdown.get(domain, 0) + 1
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


def get_last_fallback_meta() -> dict[str, object]:
    return {
        "fallback_candidate_count": _LAST_FALLBACK_META.get("fallback_candidate_count", 0),
        "fallback_valid_count": _LAST_FALLBACK_META.get("fallback_valid_count", 0),
        "fallback_domain_breakdown": dict(_LAST_FALLBACK_META.get("fallback_domain_breakdown", {})),
    }


def find_products(
    query_text: str,
    common_filters: Optional[CommonProperties],
    product_filters: Optional[ProductQuery] = None,
    web_count: int = 5,
    allow_web_fallback: bool = True,
) -> ProductSearchResults:
    global _LAST_FALLBACK_META
    repo = ProductRepository()
    query_embedding = embed_text(query_text or "")
    internal_results = repo.search_products(
        query_embedding=query_embedding,
        common_filters=common_filters,
        product_filters=product_filters,
        limit=10,
    )

    external_results: list[ProductResult] = []
    _LAST_FALLBACK_META = {
        "fallback_candidate_count": 0,
        "fallback_valid_count": 0,
        "fallback_domain_breakdown": {},
    }
    if not internal_results and allow_web_fallback:
        web_query = ((query_text or "").strip() or build_web_query(common_filters).strip() or "products")
        try:
            brave_client = BraveSearchClient.from_env()
            web_payload = brave_client.shopping_search(web_query, count=20)
            external_results, fallback_meta = _web_results_to_products(web_payload, max(1, web_count))
            _LAST_FALLBACK_META = fallback_meta
        except (ValueError, BraveSearchError):
            pass

    return ProductSearchResults(
        internal_results=internal_results,
        external_results=external_results,
    )
