from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field
from langchain_core.tools import tool

from tool_registry.adapters import (
    brave_news_search_tool as _brave_news_search_tool,
    brave_shopping_search_tool as _brave_shopping_search_tool,
    brave_web_search_tool as _brave_web_search_tool,
    find_products_tool as _find_products_tool,
    generic_web_search_tool as _generic_web_search_tool,
    get_historical_month_weather_tool as _get_historical_month_weather_tool,
    list_product_categories_tool as _list_product_categories_tool,
    resolve_city_location_tool as _resolve_city_location_tool,
)


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


class ResolveCityLocationArgs(BaseModel):
    city: str = Field(
        ...,
        description="City name only. Example: 'Toronto'. Do not include month, year, or extra context.",
    )


class HistoricalMonthWeatherArgs(BaseModel):
    city: str = Field(
        ...,
        description="City name only. Example: 'Toronto'. Do NOT include year or month.",
    )
    year: int = Field(
        ...,
        description="4-digit year. Example: 2024.",
    )
    month: int = Field(
        ...,
        description="Month number from 1 to 12. Example: 2 for February.",
        ge=1,
        le=12,
    )


class GenericWebSearchArgs(BaseModel):
    query_text: str = Field(
        ...,
        description="Search query text. Use a single string.",
    )
    search_type: Literal["web_search", "news_search", "suggestion_search"] = Field(
        default="web_search",
        description="Type of web search to run.",
    )
    country: str = Field(
        default="CA",
        description="Two-letter country code used for search localization.",
    )
    count: int = Field(
        default=5,
        description="Maximum number of results to request.",
        ge=1,
    )
    params: dict[str, Any] | None = Field(
        default=None,
        description="Optional provider-specific parameters.",
    )


class BraveNewsSearchArgs(BaseModel):
    q: str = Field(
        ...,
        description="News search query text.",
    )
    params: dict[str, Any] | None = Field(
        default=None,
        description="Optional provider-specific parameters.",
    )


class ListProductCategoriesArgs(BaseModel):
    limit: int = Field(
        default=200,
        description="Maximum number of categories to return.",
        ge=1,
    )


def _model_to_dict(value: BaseModel | dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, BaseModel):
        return value.model_dump(exclude_none=True)
    return value


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
    return _find_products_tool(
        query_text=query_text,
        common_filters=_model_to_dict(common_filters),
        product_filters=_model_to_dict(product_filters),
        web_count=web_count,
        allow_web_fallback=allow_web_fallback,
    )


@tool(
    "resolve_city_location",
    args_schema=ResolveCityLocationArgs,
    description="""
Resolve a city into normalized location metadata for weather-aware shopping decisions.

Required fields:
- city (string)

Example valid call:
{
  "city": "Toronto"
}
""",
)
def resolve_city_location(city: str):
    return _resolve_city_location_tool(city=city)

@tool(
    "get_historical_month_weather",
    args_schema=HistoricalMonthWeatherArgs,
    description="""
Get historical weather data for a specific city, year, and month. This is useful for context on the usual weather in a city.

Required fields:
- city (string)
- year (integer)
- month (integer 1-12)

Do NOT combine year and month into the city field.

Example valid call:
{
  "city": "Toronto",
  "year": 2024,
  "month": 2
}
"""
)
def get_historical_month_weather(city: str, year: int, month: int):
    return _get_historical_month_weather_tool(city=city, year=year, month=month)


@tool(
    "generic_web_search",
    args_schema=GenericWebSearchArgs,
    description="""
Run a general web, news, or suggestion search.

Required fields:
- query_text (string)

Optional fields:
- search_type (web_search | news_search | suggestion_search)
- country (string)
- count (integer)
- params (object)

Example valid call:
{
  "query_text": "best summer jackets",
  "search_type": "web_search"
}
""",
)
def generic_web_search(
    query_text: str,
    search_type: str = "web_search",
    country: str = "CA",
    count: int = 5,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _generic_web_search_tool(
        query_text=query_text,
        search_type=search_type,
        country=country,
        count=count,
        params=params,
    )

@tool(
    "brave_news_search",
    args_schema=BraveNewsSearchArgs,
    description="""
Search for current news results using Brave News Search.

Required fields:
- q (string)

Optional fields:
- params (object)

Example valid call:
{
  "q": "Toronto weather clothing news"
}
""",
)
def brave_news_search(q: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    return _brave_news_search_tool(q=q, params=params)

@tool(
    "list_product_categories",
    args_schema=ListProductCategoriesArgs,
    description="""
Return available product categories from the internal catalog.

Optional fields:
- limit (integer)

Example valid call:
{
  "limit": 200
}
""",
)
def list_product_categories(limit: int = 200) -> list[str]:
    return _list_product_categories_tool(limit=limit)


LANGCHAIN_TOOLS = [
    find_products,
    resolve_city_location,
    get_historical_month_weather,
    generic_web_search,
    brave_news_search,
    list_product_categories,
]
