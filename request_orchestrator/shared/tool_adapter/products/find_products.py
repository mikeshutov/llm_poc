from __future__ import annotations

from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from products.product_retrieval import find_products as catalog_find_products
from products.models.product_query import ProductQuery


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


@tool(
    "find_products",
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
- Do not pass tuples/arrays like ("summer clothing", "Toronto").
- If you need weather context, use the weather tools separately before calling this tool.

Examples:
- 'good dumbbells for sale online' -> 'dumbbells'
- 'any good protein powder for sale online' -> 'protein powder'
- 'vegan protein powder' -> 'vegan protein powder'

Example valid call:
{
  "query_text": "dumbbells"
}
""",
)
def find_products(
    query_text: str,
    product_filters: dict[str, Any] | None = None,
):
    return catalog_find_products(
        query_text=query_text,
        product_filters=ProductQuery(**product_filters.model_dump()) if product_filters else None,
    )
