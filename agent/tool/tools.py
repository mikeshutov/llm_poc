from dataclasses import dataclass

from agent.tool.products.find_products import find_products
from agent.tool.products.list_product_categories import list_product_categories
from agent.tool.weather.get_current_weather import get_current_weather
from agent.tool.weather.get_historical_month_weather import get_historical_month_weather
from agent.tool.weather.resolve_city_location import resolve_city_location
from agent.tool.finance.exchange_rates_lookup import exchange_rates_lookup
from agent.tool.finance.exchange_rates_time_series import exchange_rates_time_series
from agent.tool.search.generic_web_search import generic_web_search
from agent.tool.search.brave_news_search import news_search
from agent.tool.search.wikipedia_search import wikipedia_search
from agent.tool.search.structured_facts_lookup import structured_facts_lookup
from agent.tool.calendar.public_holidays_lookup import public_holidays_lookup


@dataclass
class ToolCategory:
    tools: list
    description: str


PRODUCT_TOOLS = [find_products, list_product_categories]
WEATHER_TOOLS = [resolve_city_location, get_current_weather, get_historical_month_weather]
FINANCE_TOOLS = [exchange_rates_lookup, exchange_rates_time_series]
SEARCH_TOOLS = [generic_web_search, news_search, wikipedia_search, structured_facts_lookup]
CALENDAR_TOOLS = [public_holidays_lookup]

TOOL_CATEGORIES: dict[str, ToolCategory] = {
    "products": ToolCategory(tools=PRODUCT_TOOLS, description="Search and browse products and product categories from the catalog."),
    "weather": ToolCategory(tools=WEATHER_TOOLS, description="Look up current or historical weather conditions for a city."),
    "finance": ToolCategory(tools=FINANCE_TOOLS, description="Retrieve currency exchange rates and historical rate time series."),
    "search": ToolCategory(tools=SEARCH_TOOLS, description="Search the web, news, Wikipedia, or structured knowledge for general information."),
    "calendar": ToolCategory(tools=CALENDAR_TOOLS, description="Look up public holidays for a country and year."),
}

tools = [*PRODUCT_TOOLS, *WEATHER_TOOLS, *FINANCE_TOOLS, *SEARCH_TOOLS, *CALENDAR_TOOLS]
