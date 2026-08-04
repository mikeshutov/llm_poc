from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from products.product_retrieval import find_products_web as web_find_products


class FindProductsWebArgs(BaseModel):
    query_text: str = Field(
        ...,
        description="Short product search query. Example: 'summer clothing'. Use a single string, not a tuple or list.",
    )
    count: int = Field(
        default=5,
        description="Maximum number of results to return.",
        ge=1,
    )


@tool(
    "find_products_web",
    args_schema=FindProductsWebArgs,
    description="""
Search the web for products when the internal catalog has no results. Or when external searches are requested.

Required fields:
- query_text (string)

Optional fields:
- count (integer)

Example valid call:
{
  "query_text": "men's casual shirts"
}
""",
)
def find_products_web(
    query_text: str,
    count: int = 5,
):
    return web_find_products(query_text=query_text, count=count)
