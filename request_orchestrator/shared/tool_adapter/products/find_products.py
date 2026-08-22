from __future__ import annotations

from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from products.models.product_result import ProductResult
from products.models.product_search_results import ProductSearchResults
from products.product_retrieval import find_products as catalog_find_products
from products.models.product_query import ProductQuery
from request_orchestrator.models.evidence import EvidenceUrl, EvidenceView, HydratedEvidence, ToolResult
from tool.constants import TOOL_NAME_FIND_PRODUCTS
from tool.constants import TOOL_RESULT_TYPE_PRODUCT_RESULTS


class ProductFiltersArgs(BaseModel):
    category: list[str] | str | None = Field(default=None, description="Optional category filter. One or more values. Prefer categories returned by list_product_categories.")
    style: str | None = Field(default=None, description="Optional style filter. Example: 'casual'.")
    color: str | None = Field(default=None, description="Optional color filter. Example: 'black'.")
    price_min: float | None = Field(default=None, description="Optional minimum price.")
    price_max: float | None = Field(default=None, description="Optional maximum price.")
    gender: str | None = Field(default=None, description="Optional gender filter. Example: 'Men' or 'Women'.")


class FindProductsArgs(BaseModel):
    query_text: str = Field(
        ...,
        description=(
            "Core product noun phrase only, usually just the item name. "
            "Use the user's wording as closely as possible, but strip generic shopping words and quality words. "
            "Do not expand with speculative synonyms, subtypes, or extra descriptors unless the user explicitly asked for them."
        ),
    )
    product_filters: ProductFiltersArgs | None = Field(
        default=None,
        description="Optional filters: category, style, color, price_min, price_max, gender.",
    )


class ProductEvidenceMetadata(BaseModel):
    category: str | None = None
    color: str | None = None
    style: str | None = None
    gender: str | None = None
    season: str | None = None
    year: int | None = None
    price: float | None = None
    score: float | None = None
    product_source: str
    retrieved_count: int
    reranked: bool


def _product_summary(product: ProductResult) -> str:
    parts: list[str] = []
    if product.description:
        parts.append(product.description.strip())
    if product.price is not None:
        parts.append(f"Price {product.price}")
    return ". ".join(part for part in parts if part) or f"Product result for {product.name}."


def _tool_result(result: ProductSearchResults) -> ToolResult:
    hydrated_evidence: list[HydratedEvidence] = []
    evidence_views: list[EvidenceView] = []
    for product in [*result.internal_results, *result.external_results]:
        url = (product.url or "").strip()
        metadata = ProductEvidenceMetadata(
            category=product.category,
            color=product.color,
            style=product.style,
            gender=product.gender,
            season=product.season,
            year=product.year,
            price=product.price,
            score=product.score,
            product_source=product.source.value,
            retrieved_count=result.retrieved_count,
            reranked=result.reranked,
        )
        hydrated = HydratedEvidence(
            item_id=product.id,
            tool_name=TOOL_NAME_FIND_PRODUCTS,
            title=product.name,
            summary=_product_summary(product),
            urls=[EvidenceUrl(url=url, url_type="website")] if url else [],
            image_url=(product.image_url or "").strip(),
            source=TOOL_NAME_FIND_PRODUCTS,
            entity_type=TOOL_RESULT_TYPE_PRODUCT_RESULTS,
            metadata=metadata.model_dump(exclude_none=True),
            raw_payload=product,
        )
        hydrated_evidence.append(hydrated)
        evidence_views.append(
            EvidenceView(
                item_id=hydrated.item_id,
                title=hydrated.title,
                summary=hydrated.summary,
                metadata=dict(hydrated.metadata),
            )
        )
    return ToolResult(result=result, evidence_views=evidence_views, hydrated_evidence=hydrated_evidence)


@tool(
    TOOL_NAME_FIND_PRODUCTS,
    args_schema=FindProductsArgs,
    description="""
Search the internal product catalog.

Required fields:
- query_text (string)
- a short core product phrase and nothing more

Optional fields:
- product_filters (object): category, style, color, price_min, price_max, gender

Important:
- Pass a single object with named fields.
- query_text should usually just be the item name or core noun phrase.
- Remove generic shopping words like 'buy', 'for sale', 'online'.
- Remove generic quality words like 'good', 'best', 'top', 'nice'.
- Keep explicit user constraints that change the product itself.
- Do not expand query_text with guessed synonyms, categories, product variants, ingredients, or audiences.

Examples:
- 'good dumbbells for sale online' -> 'dumbbells'
- 'vegan protein powder' -> 'vegan protein powder'

Example valid call:
{
  "query_text": "dumbbells"
}
""",
)
def find_products(
    query_text: str,
    product_filters: ProductFiltersArgs | None = None,
) -> ToolResult:
    return _tool_result(catalog_find_products(
        query_text=query_text,
        product_filters=ProductQuery(**product_filters.model_dump()) if product_filters else None,
    ))
