from __future__ import annotations

from typing import Any

from personalization.profile.models import UserProfile
from reranker import Candidate, DEFAULT_TOP_K, rerank_candidates
from products.models.product_result import ProductResult


def product_result_to_candidate(product: ProductResult) -> Candidate:
    metadata = {
        "source": product.source.value,
    }
    if product.score is not None:
        metadata["retrieval_distance"] = product.score

    return Candidate(
        id=product.id,
        candidate_type="product",
        title=product.name,
        content={
            "name": product.name,
            "description": product.description,
            "url": product.url,
            "image_url": product.image_url,
        },
        attributes={
            "category": product.category,
            "color": product.color,
            "style": product.style,
            "gender": product.gender,
            "season": product.season,
            "year": product.year,
            "price": product.price,
        },
        metadata=metadata,
    )


def prepare_product_candidates(products: list[ProductResult]) -> list[Candidate]:
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
