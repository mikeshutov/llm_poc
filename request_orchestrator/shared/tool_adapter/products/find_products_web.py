from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from products.models.product_result import ProductResult
from products.models.product_search_results import ProductSearchResults
from products.product_retrieval import find_products_web as web_find_products
from request_orchestrator.models.evidence import EvidenceUrl, EvidenceUrlType, EvidenceView, HydratedEvidence, ToolResult
from tool.constants import TOOL_NAME_FIND_PRODUCTS_WEB
from tool.constants import TOOL_RESULT_TYPE_PRODUCT_RESULTS


class FindProductsWebArgs(BaseModel):
    query_text: str = Field(
        ...,
        description=(
            "Core product noun phrase only, usually just the item name. "
            "Use the user's wording as closely as possible, but strip generic shopping words and quality words. "
            "Do not expand with speculative synonyms, product subtypes, ingredients, brands, geographies, "
            "or extra buying-intent phrases unless the user explicitly asked for them. "
            "Example: 'good dumbbells for sale online' -> 'dumbbells'; "
        ),
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
            tool_name=TOOL_NAME_FIND_PRODUCTS_WEB,
            title=product.name,
            summary=_product_summary(product),
            urls=[EvidenceUrl(url=url, url_type=EvidenceUrlType.WEBSITE)] if url else [],
            image_url=(product.image_url or "").strip(),
            source=TOOL_NAME_FIND_PRODUCTS_WEB,
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
    TOOL_NAME_FIND_PRODUCTS_WEB,
    args_schema=FindProductsWebArgs,
    description="""
Search the web for products when the internal catalog has no results, or when external product searches are requested.

Important query rules:
- query_text should usually just be the item name or core noun phrase.
- Remove generic shopping words like 'buy', 'for sale', 'online'.
- Remove generic quality words like 'good', 'best', 'top', 'nice'.
- Keep explicit user constraints that change the product itself.
- Do not expand the query with guessed variants, ingredients, subtypes, audiences, brands, or locations.
- Do not add terms like countries, brands, whey, isolate, casein, vegan, etc. unless the user explicitly asked for them.
- Prefer the minimal product phrase over keyword stuffing.
- Result count is fixed internally.
- Do not pass count or other limit fields.

Examples:
- 'good dumbbells for sale online' -> 'dumbbells'
- 'vegan protein powder' -> 'vegan protein powder'

Example valid call:
{
  "query_text": "dumbbells"
}
""",
)
def find_products_web(
    query_text: str,
) -> ToolResult:
    return _tool_result(web_find_products(query_text=query_text))
