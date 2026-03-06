from __future__ import annotations

from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from agent.tool.common import model_to_dict
from integrations.brave import find_products as _find_products
from integrations.models.common_properties import CommonProperties
from products.models.product_query import ProductQuery


class CommonFiltersArgs(BaseModel):
    color: str | None = Field(default=None, description="Optional color filter. Example: 'black'.")
    price_min: float | None = Field(default=None, description="Optional minimum price.")
    price_max: float | None = Field(default=None, description="Optional maximum price.")
    gender: str | None = Field(
        default=None,
        description="Optional gender filter. Example: 'Men' or 'Women'.",
    )


class ProductFiltersArgs(BaseModel):
    category: str | None = Field(
        default=None,
        description="Optional category filter. Prefer categories returned by list_product_categories.",
    )
    style: str | None = Field(default=None, description="Optional style filter. Example: 'casual'.")


class FindProductsArgs(BaseModel):
    query_text: str = Field(
        ...,
        description="Short product search query. Example: 'summer clothing'. Use a single string, not a tuple or list.",
    )
    common_filters: CommonFiltersArgs | None = Field(
        default=None,
        description="Optional shared filters like color and price.",
    )
    product_filters: ProductFiltersArgs | None = Field(
        default=None,
        description="Optional product-specific filters like category and style.",
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
- common_filters (object)
- product_filters (object)
- web_count (integer)
- allow_web_fallback (boolean)

Important:
- Pass a single object with named fields.
- Do not pass tuples like ("summer clothing", "Toronto").
- If you need weather context, use the weather tools separately before calling this tool.

Example valid call:
{
  "query_text": "summer clothing",
  "product_filters": {
    "category": "Shirts"
  }
}
""",
)
def find_products(
    query_text: str,
    common_filters: dict[str, Any] | None = None,
    product_filters: dict[str, Any] | None = None,
    web_count: int = 5,
    allow_web_fallback: bool = True,
):
    common_dict = model_to_dict(common_filters)
    product_dict = model_to_dict(product_filters)
    return _find_products(
        query_text=query_text,
        common_filters=CommonProperties(**common_dict) if common_dict else None,
        product_filters=ProductQuery(**product_dict) if product_dict else None,
        web_count=web_count,
        allow_web_fallback=allow_web_fallback,
    )
