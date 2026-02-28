from __future__ import annotations

from typing import Any

from tool_registry.adapters import (
    brave_news_search_tool,
    brave_shopping_search_tool,
    brave_suggest_search_tool,
    brave_web_search_tool,
    find_products_tool,
    generic_web_search_tool,
    get_historical_month_weather_tool,
    list_product_categories_tool,
    resolve_city_location_tool,
)
from tool_registry.registry import ToolRegistry

GLOBAL_TOOL_REGISTRY = ToolRegistry()

_REGISTERED_DEFAULTS = False


def register_default_tools() -> None:
    global _REGISTERED_DEFAULTS
    if _REGISTERED_DEFAULTS:
        return

    GLOBAL_TOOL_REGISTRY.register("find_products", find_products_tool)
    GLOBAL_TOOL_REGISTRY.register("resolve_city_location", resolve_city_location_tool)
    GLOBAL_TOOL_REGISTRY.register("get_historical_month_weather", get_historical_month_weather_tool)
    GLOBAL_TOOL_REGISTRY.register("generic_web_search", generic_web_search_tool)
    GLOBAL_TOOL_REGISTRY.register("brave_web_search", brave_web_search_tool)
    GLOBAL_TOOL_REGISTRY.register("brave_news_search", brave_news_search_tool)
    GLOBAL_TOOL_REGISTRY.register("brave_suggest_search", brave_suggest_search_tool)
    GLOBAL_TOOL_REGISTRY.register("brave_shopping_search", brave_shopping_search_tool)
    GLOBAL_TOOL_REGISTRY.register("list_product_categories", list_product_categories_tool)
    _REGISTERED_DEFAULTS = True


def call_tool(name: str, params: dict[str, Any] | None = None) -> Any:
    register_default_tools()
    return GLOBAL_TOOL_REGISTRY.call_tool(name=name, params=params)
