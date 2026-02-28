from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any
from langchain_core.tools import tool
from intent_layer.models.parsed_request import QueryDetails
from intent_processing.generic_search import generic_web_search
from products.models.product_query import ProductQuery
from websearch.domain.product_retrieval import find_products
from products.repository.product_repository import ProductRepository
from websearch.clients.brave_client import BraveSearchClient
from websearch.models.common_properties import CommonProperties
from websearch.models.search_type import SearchType
from websearch.models.web_search_params import WebSearchParams
from weather.clients.open_meteo_client import OpenMeteoClient

_weather_client = OpenMeteoClient()


def find_products_tool(
    query_text: str,
    common_filters: dict[str, Any] | None = None,
    product_filters: dict[str, Any] | None = None,
    web_count: int = 5,
    allow_web_fallback: bool = True,
):
    resolved_filters = CommonProperties(**common_filters) if common_filters else None
    resolved_product_filters = ProductQuery(**product_filters) if product_filters else None
    return find_products(
        query_text=query_text,
        common_filters=resolved_filters,
        product_filters=resolved_product_filters,
        web_count=web_count,
        allow_web_fallback=allow_web_fallback,
    )


def resolve_city_location_tool(city: str):
    result = _weather_client.geocode_city(city)
    return asdict(result) if is_dataclass(result) else dict(result)


def get_historical_month_weather_tool(city: str, year: int, month: int):
    result = _weather_client.get_historical_month(city, year, month)
    payload = asdict(result) if is_dataclass(result) else dict(result)
    payload.pop("daily", None)
    return payload


def _coerce_search_type(search_type: str) -> SearchType:
    try:
        return SearchType(search_type)
    except ValueError as exc:
        allowed = ", ".join(t.value for t in SearchType)
        raise ValueError(f"Invalid search_type '{search_type}'. Allowed values: {allowed}.") from exc


def generic_web_search_tool(
    query_text: str,
    search_type: str = "web_search",
    country: str = "CA",
    count: int = 5,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    details = QueryDetails(
        query_text=query_text,
        search_type=_coerce_search_type(search_type),
    )
    return generic_web_search(
        details,
        country=country,
        count=count,
        **(params or {}),
    )


def brave_web_search_tool(
    q: str,
    count: int = 10,
    offset: int = 0,
    country: str = "CA",
    search_lang: str = "en",
    extra_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    client = BraveSearchClient.from_env()
    return client.web_search(
        WebSearchParams(
            q=q,
            count=count,
            offset=offset,
            country=country,
            search_lang=search_lang,
            extra_params=extra_params or {},
        )
    )


def brave_news_search_tool(q: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    client = BraveSearchClient.from_env()
    return client.news_search(q, **(params or {}))


def brave_suggest_search_tool(
    q: str,
    country: str = "CA",
    count: int = 5,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    client = BraveSearchClient.from_env()
    return client.suggest(q, country=country, count=count, **(params or {}))


def brave_shopping_search_tool(
    q: str,
    sources: list[str] | None = None,
    count: int = 10,
    offset: int = 0,
    country: str = "US",
    search_lang: str = "en",
    extra_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    client = BraveSearchClient.from_env()
    result = client.shopping_search(
        q=q,
        sources=sources,
        count=count,
        offset=offset,
        country=country,
        search_lang=search_lang,
        **(extra_params or {}),
    )
    return {
        "query": result.query,
        "sources": result.sources,
        "results": result.results,
        "raw": result.raw,
    }


def list_product_categories_tool(limit: int = 200) -> list[str]:
    repo = ProductRepository()
    return repo.list_categories(limit=limit)
