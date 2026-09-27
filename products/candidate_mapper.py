from __future__ import annotations

from typing import Any

from personalization.profile.models import UserProfile
from reranker import DEFAULT_TOP_K, RerankerCandidate, rerank_candidates
from products.models.product_result import ProductResult


def product_result_to_candidate(product: ProductResult) -> RerankerCandidate:
    fields = [
        f"title: {product.name}",
        *(
            [f"description: {product.description}"]
            if product.description
            else []
        ),
        *(
            [f"category: {product.category}"]
            if product.category
            else []
        ),
        *(
            [f"color: {product.color}"]
            if product.color
            else []
        ),
        *(
            [f"style: {product.style}"]
            if product.style
            else []
        ),
        *(
            [f"gender: {product.gender}"]
            if product.gender
            else []
        ),
        *(
            [f"season: {product.season}"]
            if product.season
            else []
        ),
        *(
            [f"year: {product.year}"]
            if product.year is not None
            else []
        ),
        *(
            [f"price: {product.price}"]
            if product.price is not None
            else []
        ),
    ]
    return RerankerCandidate(
        id=product.id,
        text="\n".join(fields),
    )


def prepare_product_candidates(products: list[ProductResult]) -> list[RerankerCandidate]:
    return [product_result_to_candidate(product) for product in products]


def rerank_product_results(
    products: list[ProductResult],
    *,
    goal: str | None = None,
    query: str | None = None,
    user_profile: UserProfile | None = None,
    llm: Any | None = None,
) -> list[ProductResult]:
    if len(products) <= 1:
        return list(products)[:DEFAULT_TOP_K]

    resolved_goal = goal if goal is not None else query
    candidates = prepare_product_candidates(products)
    ranked_candidates = rerank_candidates(candidates, goal=resolved_goal, user_profile=user_profile, llm=llm)
    products_by_id = {product.id: product for product in products}
    ranked_products: list[ProductResult] = []
    seen_ids: set[str] = set()

    for candidate in ranked_candidates:
        product = products_by_id.get(candidate.id)
        if product is None or product.id in seen_ids:
            continue
        ranked_products.append(product)
        seen_ids.add(product.id)

    return ranked_products[:DEFAULT_TOP_K]
