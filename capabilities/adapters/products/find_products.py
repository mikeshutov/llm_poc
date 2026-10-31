from __future__ import annotations

from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from common.html_text import html_to_plain_text
from products.models.product_result import ProductResult
from products.models.product_search_results import ProductSearchResults
from products.product_retrieval import find_products as catalog_find_products
from products.product_retrieval import find_products_web as web_find_products
from products.models.product_query import ProductQuery
from retrieval.evidence import EvidenceUrl, EvidenceUrlType, EvidenceView, ToolMetadata, ToolResult
from capabilities.registry.constants import TOOL_NAME_FIND_PRODUCTS
from capabilities.registry.constants import TOOL_RESULT_TYPE_PRODUCT_RESULTS


class ProductFiltersArgs(BaseModel):
    category: list[str] | str | None = Field(default=None, description="Optional category filter. One or more values. Prefer categories returned by list_product_categories.")
    style: str | None = Field(default=None, description="Optional style filter. Example: 'casual'.")
    color: str | None = Field(default=None, description="Optional color filter. Example: 'black'.")
    price_min: float | None = Field(default=None, description="Optional minimum price.")
    price_max: float | None = Field(default=None, description="Optional maximum price.")
    gender: str | None = Field(default=None, description="Optional gender filter. Example: 'Men' or 'Women'.")


class FindProductsArgs(BaseModel):
    source: Literal["internal", "web", "both"] = Field(
        ...,
        description="Product source to query: internal catalog, external web shopping, or both.",
    )
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


def _product_summary(product: ProductResult) -> str:
    parts: list[str] = []
    if product.description:
        parts.append(html_to_plain_text(product.description))
    if product.price is not None:
        parts.append(f"Price {product.price}")
    return ". ".join(part for part in parts if part) or f"Product result for {product.name}."


def _tool_result(result: ProductSearchResults) -> ToolResult:
    evidence: list[EvidenceView] = []
    products = [*result.internal_results, *result.external_results]
    product_sources = sorted({product.source.value for product in products})
    tool_metadata = ToolMetadata(
        retrieved_count=result.retrieved_count,
        reranked=result.reranked,
        product_source=(
            product_sources[0] if len(product_sources) == 1 else product_sources
        )
        if product_sources
        else None,
    )

    for product in products:
        url = (product.url or "").strip()
        metadata = ProductEvidenceMetadata(
            category=product.category,
            color=product.color,
            style=product.style,
            gender=product.gender,
            season=product.season,
            year=product.year,
            price=product.price,
        )
        evidence_view = EvidenceView(
            item_id=product.id,
            tool_name=TOOL_NAME_FIND_PRODUCTS,
            title=product.name,
            summary=_product_summary(product),
            urls=[EvidenceUrl(url=url, url_type=EvidenceUrlType.WEBSITE)] if url else [],
            image_url=(product.image_url or "").strip(),
            source=TOOL_NAME_FIND_PRODUCTS,
            entity_type=TOOL_RESULT_TYPE_PRODUCT_RESULTS,
            llm_metadata=metadata.model_dump(exclude_none=True),
            raw_payload=product,
        )
        evidence.append(evidence_view)
    return ToolResult(result=result, tool_metadata=tool_metadata, evidence=evidence)


@tool(
    TOOL_NAME_FIND_PRODUCTS,
    args_schema=FindProductsArgs,
    description="""
Search for products using the requested product source.

Use source='internal' for the internal catalog, source='web' for external shopping results, or source='both' to combine both result sets.



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

""",
)
def search_products(
    source: Literal["internal", "web", "both"],
    query_text: str,
    product_filters: ProductFiltersArgs | None = None,
) -> ToolResult:
    if source == "web" and product_filters is not None:
        raise ValueError("product_filters can only be used with source='internal' or source='both'.")

    internal_results: list[ProductResult] = []
    external_results: list[ProductResult] = []
    if source in {"internal", "both"}:
        internal_results = catalog_find_products(
            query_text=query_text,
            product_filters=ProductQuery(**product_filters.model_dump()) if product_filters else None,
        ).internal_results
    if source in {"web", "both"}:
        external_results = web_find_products(query_text=query_text).external_results

    return _tool_result(
        ProductSearchResults(
            internal_results=internal_results,
            external_results=external_results,
            retrieved_count=len(internal_results) + len(external_results),
            reranked=True,
        )
    )
