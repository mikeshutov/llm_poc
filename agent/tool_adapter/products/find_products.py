from __future__ import annotations

from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from products.product_retrieval import find_products as _find_products
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
        description="Short product search query. Example: 'summer clothing'. Use a single string, not a tuple or list.",
    )
    product_filters: ProductFiltersArgs | None = Field(
        default=None,
        description="Optional filters: category, style, color, price_min, price_max, gender.",
    )
    web_count: int = Field(
        default=5,
        description="Maximum number of web fallback results when fallback is allowed.",
        ge=1,
    )
    allow_web_fallback: bool = Field(
        default=True,
        description="Whether web fallback is allowed if local catalog results are insufficient.",
    )


@tool(
    "find_products",
    args_schema=FindProductsArgs,
    description="""
Search the local product catalog first and optionally fall back to web results.

Required fields:
- query_text (string)

Optional fields:
- product_filters (object): category, style, color, price_min, price_max, gender
- web_count (integer)
- allow_web_fallback (boolean)

Important:
- Pass a single object with named fields.
- Do not pass tuples like ("summer clothing", "Toronto").
- If you need weather context, use the weather tools separately before calling this tool.

Example valid call:
{
  "query_text": "men's clothing",
  "product_filters": {
    "gender": "Men",
    "price_max": 40,
    "category": "Shirts"
  }
}
""",
)
def find_products(
    query_text: str,
    product_filters: dict[str, Any] | None = None,
    web_count: int = 5,
    allow_web_fallback: bool = True,
):
    return _find_products(
        query_text=query_text,
        product_filters=ProductQuery(**product_filters.model_dump()) if product_filters else None,
        web_count=web_count,
        allow_web_fallback=allow_web_fallback,
    )
