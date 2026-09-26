from products.candidate_mapper import (
    prepare_product_candidates,
    product_result_to_candidate,
    rerank_product_results,
)
from products.models.product_result import ProductResult
from products.models.product_search_results import ProductSearchResults
from products.models.product_source import ProductSource
from products.repository.product_repository import ProductRepository
from reranker.constants import DEFAULT_TOP_K
from test_utilities.mock_llm import MockLLM


def test_product_result_to_candidate_maps_product_fields() -> None:
    product = ProductResult(
        id="sku-123",
        name="Trail Running Shoes",
        description="Men Blue Footwear Sport Spring. Year: 2026",
        category="Footwear",
        color="Blue",
        style="Sport",
        gender="Men",
        season="Spring",
        year=2026,
        price=129.99,
        url="https://example.com/products/sku-123",
        image_url="https://example.com/products/sku-123.jpg",
        score=0.87,
        source=ProductSource.DB,
    )

    candidate = product_result_to_candidate(product)

    assert candidate.id == "sku-123"
    assert candidate.candidate_type == "product"
    assert candidate.title == "Trail Running Shoes"
    assert candidate.content["description"] == "Men Blue Footwear Sport Spring. Year: 2026"
    assert candidate.content["url"] == "https://example.com/products/sku-123"
    assert candidate.attributes["price"] == 129.99
    assert candidate.metadata["source"] == "db"
    assert candidate.metadata["retrieval_distance"] == 0.87
    assert candidate.score is None


def test_prepare_product_candidates_maps_multiple_products() -> None:
    products = [
        ProductResult(
            id="sku-1",
            name="Product One",
            description="First product description",
            category=None,
            color=None,
            style=None,
            gender=None,
            season=None,
            year=None,
            price=None,
            source=ProductSource.WEB,
        ),
        ProductResult(
            id="sku-2",
            name="Product Two",
            description="Second product description",
            category=None,
            color=None,
            style=None,
            gender=None,
            season=None,
            year=None,
            price=None,
            source=ProductSource.DB,
        ),
    ]

    candidates = prepare_product_candidates(products)

    assert [candidate.id for candidate in candidates] == ["sku-1", "sku-2"]
    assert candidates[0].content["description"] == "First product description"
    assert all(candidate.candidate_type == "product" for candidate in candidates)


def test_product_search_results_exposes_retrieval_metadata() -> None:
    product = ProductResult(
        id="sku-1",
        name="Product One",
        description="First product description",
        category=None,
        color=None,
        style=None,
        gender=None,
        season=None,
        year=None,
        price=None,
        source=ProductSource.DB,
    )

    results = ProductSearchResults(
        internal_results=[product],
        external_results=[],
        retrieved_count=8,
        reranked=True,
    )

    assert results.retrieved_count == 8
    assert results.reranked is True


def test_product_repository_uses_null_description_when_column_missing() -> None:
    repo = ProductRepository.__new__(ProductRepository)
    repo._has_description_column = False

    sql, _ = repo._build_search_sql([0.1, 0.2], None, 6)

    assert "NULL AS description" in sql
    assert "id, name," in sql


def test_product_repository_selects_description_when_column_exists() -> None:
    repo = ProductRepository.__new__(ProductRepository)
    repo._has_description_column = True

    sql, _ = repo._build_search_sql([0.1, 0.2], None, 6)

    assert "name, description," in sql


def test_rerank_product_results_skips_llm_when_result_count_is_at_or_below_top_k() -> None:
    llm = MockLLM([
        '{"ranked_ids": ["sku-2", "sku-1"]}'
    ])
    products = [
        ProductResult(
            id="sku-1",
            name="Product One",
            description="First product description",
            category=None,
            color=None,
            style=None,
            gender=None,
            season=None,
            year=None,
            price=None,
            source=ProductSource.DB,
        ),
        ProductResult(
            id="sku-2",
            name="Product Two",
            description="Second product description",
            category=None,
            color=None,
            style=None,
            gender=None,
            season=None,
            year=None,
            price=None,
            source=ProductSource.DB,
        ),
    ]

    ranked_products = rerank_product_results(products, query="best product", llm=llm)

    assert [product.id for product in ranked_products] == ["sku-1", "sku-2"]
    assert llm.last_prompt is None


def test_rerank_product_results_returns_only_default_top_k_products() -> None:
    product_ids = [f"sku-{index}" for index in range(1, 9)]
    llm = MockLLM([
        '{"ranked_ids": ["sku-8", "sku-7", "sku-6", "sku-5", "sku-4", "sku-3", "sku-2", "sku-1"]}'
    ])
    products = [
        ProductResult(
            id=product_id,
            name=product_id,
            description=f"Description for {product_id}",
            category=None,
            color=None,
            style=None,
            gender=None,
            season=None,
            year=None,
            price=None,
            source=ProductSource.DB,
        )
        for product_id in product_ids
    ]

    ranked_products = rerank_product_results(products, llm=llm)

    assert len(ranked_products) == DEFAULT_TOP_K
    assert [product.id for product in ranked_products] == [
        "sku-8",
        "sku-7",
        "sku-6",
        "sku-5",
        "sku-4",
        "sku-3",
    ]
