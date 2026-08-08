from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from products.product_retrieval import find_products_web as web_find_products


class FindProductsWebArgs(BaseModel):
    query_text: str = Field(
        ...,
        description=(
            "Core product noun phrase only, usually just the item name. "
            "Use the user's wording as closely as possible, but strip generic shopping words and quality words. "
            "Do not expand with speculative synonyms, product subtypes, ingredients, brands, geographies, "
            "or extra buying-intent phrases unless the user explicitly asked for them. "
            "Examples: 'good dumbbells for sale online' -> 'dumbbells'; "
            "'any good protein powder for sale online' -> 'protein powder'."
        ),
    )


@tool(
    "find_products_web",
    args_schema=FindProductsWebArgs,
    description="""
Search the web for products when the internal catalog has no results, or when external product searches are requested.

Required fields:
- query_text (string)

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
- 'any good protein powder for sale online' -> 'protein powder'
- 'vegan protein powder' -> 'vegan protein powder'

Example valid call:
{
  "query_text": "dumbbells"
}
""",
)
def find_products_web(
    query_text: str,
):
    return web_find_products(query_text=query_text)
