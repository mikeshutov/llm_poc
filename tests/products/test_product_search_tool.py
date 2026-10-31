import pytest

from products.models.product_result import ProductResult
from products.models.product_search_results import ProductSearchResults
from products.models.product_source import ProductSource
from capabilities.adapters.products.find_products import search_products


def _internal_result() -> ProductSearchResults:
    return ProductSearchResults(
        internal_results=[
            ProductResult(
                id="internal-1",
                name="Internal shoes",
                description="Catalog result",
                category="Footwear",
                color=None,
                style=None,
                gender=None,
                season=None,
                year=None,
                price=100,
                source=ProductSource.DB,
            )
        ],
        external_results=[],
        retrieved_count=1,
        reranked=True,
    )


def _web_result() -> ProductSearchResults:
    return ProductSearchResults(
        internal_results=[],
        external_results=[
            ProductResult(
                id="https://example.com/shoes",
                name="Web shoes",
                description="Web result",
                category=None,
                color=None,
                style=None,
                gender=None,
                season=None,
                year=None,
                price=120,
                url="https://example.com/shoes",
                source=ProductSource.WEB,
            )
        ],
        retrieved_count=1,
        reranked=True,
    )


def test_product_search_supports_explicit_internal_web_and_both_sources(monkeypatch) -> None:
    monkeypatch.setitem(search_products.func.__globals__, "catalog_find_products", lambda **_: _internal_result())
    monkeypatch.setitem(search_products.func.__globals__, "web_find_products", lambda **_: _web_result())

    internal = search_products.invoke({"source": "internal", "query_text": "shoes"})
    web = search_products.invoke({"source": "web", "query_text": "shoes"})
    both = search_products.invoke({"source": "both", "query_text": "shoes"})

    assert [item.name for item in internal.result.internal_results] == ["Internal shoes"]
    assert web.result.internal_results == []
    assert [item.name for item in web.result.external_results] == ["Web shoes"]
    assert [item.name for item in both.result.internal_results] == ["Internal shoes"]
    assert [item.name for item in both.result.external_results] == ["Web shoes"]
    assert both.result.retrieved_count == 2
    assert {item.source for item in both.result.internal_results + both.result.external_results} == {
        ProductSource.DB,
        ProductSource.WEB,
    }


def test_product_search_rejects_filters_for_web_only(monkeypatch) -> None:
    with pytest.raises(ValueError, match="product_filters"):
        search_products.invoke(
            {
                "source": "web",
                "query_text": "shoes",
                "product_filters": {"category": "Footwear"},
            }
        )
