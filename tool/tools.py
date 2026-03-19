from dataclasses import dataclass, field

from agent.tool_adapter.products.find_products import find_products
from agent.tool_adapter.products.list_product_categories import list_product_categories
from agent.tool_adapter.weather.get_current_weather import get_current_weather
from agent.tool_adapter.weather.get_historical_month_weather import get_historical_month_weather
from agent.tool_adapter.weather.resolve_city_location import resolve_city_location
from agent.tool_adapter.finance.exchange_rates_lookup import exchange_rates_lookup
from agent.tool_adapter.finance.exchange_rates_time_series import exchange_rates_time_series
from agent.tool_adapter.finance.crypto_markets import get_crypto_markets
from agent.tool_adapter.finance.latest_exchange_rates import get_latest_exchange_rates
from agent.tool_adapter.search.generic_web_search import generic_web_search
from agent.tool_adapter.search.brave_news_search import news_search
from agent.tool_adapter.search.wikipedia_search import wikipedia_search
from agent.tool_adapter.search.structured_facts_lookup import structured_facts_lookup
from agent.tool_adapter.calendar.public_holidays_lookup import public_holidays_lookup
from agent.tool_adapter.location.get_caller_location import get_caller_location
from agent.tool_adapter.books.search_books import search_books
from agent.tool_adapter.news.hn_search import hn_search
from agent.tool_adapter.language.define_word import define_word
from agent.tool_adapter.food.search_meals import search_meals
from agent.tool_adapter.food.search_cocktails import search_cocktails
from agent.tool_adapter.fun.get_advice import get_advice
from agent.tool_adapter.fun.get_quote import get_quote
from agent.tool_adapter.fun.astronomy_picture import get_astronomy_picture
from agent.tool_adapter.search.country_lookup import country_lookup
from agent.tool_adapter.calendar.world_time import get_world_time


@dataclass
class ToolCategory:
    tools: list
    description: str
    rules: list[str] = field(default_factory=list)


PRODUCT_TOOLS = [find_products, list_product_categories]
WEATHER_TOOLS = [resolve_city_location, get_current_weather, get_historical_month_weather]
FINANCE_TOOLS = [exchange_rates_lookup, exchange_rates_time_series, get_crypto_markets, get_latest_exchange_rates]
SEARCH_TOOLS = [generic_web_search, news_search, wikipedia_search, structured_facts_lookup, hn_search, country_lookup]
CALENDAR_TOOLS = [public_holidays_lookup, get_world_time]
LOCATION_TOOLS = [get_caller_location]
BOOKS_TOOLS = [search_books]
LANGUAGE_TOOLS = [define_word]
FOOD_TOOLS = [search_meals, search_cocktails]
FUN_TOOLS = [get_advice, get_quote, get_astronomy_picture]

# if this were to grow much larger I would probably create sub categories or a tree structure of tools
TOOL_CATEGORIES: dict[str, ToolCategory] = {
    "products": ToolCategory(
        tools=PRODUCT_TOOLS,
        description="Search and browse products and product categories from the catalog.",
        rules=["For product searches utilize internal tools first before web searches."
               "Make sure that previous context is taken into account when providing filters unless explicitely told not to."],
    ),
    "weather": ToolCategory(
        tools=WEATHER_TOOLS,
        description="Look up current or historical weather conditions for a city.",
    ),
    "finance": ToolCategory(
        tools=FINANCE_TOOLS,
        description="Retrieve currency exchange rates, historical rate time series, and live cryptocurrency prices.",
    ),
    "search": ToolCategory(
        tools=SEARCH_TOOLS,
        description="Search the web, news, Wikipedia, structured knowledge, or country information for general information.",
        rules=["If you use Brave/WebSearch tools, use at most ONE of them in the entire plan."],
    ),
    "calendar": ToolCategory(
        tools=CALENDAR_TOOLS,
        description="Look up public holidays for a country and year, or get the current time for a timezone.",
    ),
    "location": ToolCategory(
        tools=LOCATION_TOOLS,
        description="Resolve the caller's geographic location by IP address.",
    ),
    "books": ToolCategory(
        tools=BOOKS_TOOLS,
        description="Search the Open Library catalog for books by title, author, or subject.",
    ),
    "language": ToolCategory(
        tools=LANGUAGE_TOOLS,
        description="Look up word definitions, meanings, parts of speech, synonyms, and antonyms.",
    ),
    "food": ToolCategory(
        tools=FOOD_TOOLS,
        description="Search for meal recipes by name or keyword, including ingredients and instructions.",
    ),
    "fun": ToolCategory(
        tools=FUN_TOOLS,
        description="Retrieve fun or interesting content: advice slips, number trivia, inspirational quotes, and NASA astronomy pictures.",
    ),
}

tools = [*PRODUCT_TOOLS, *WEATHER_TOOLS, *FINANCE_TOOLS, *SEARCH_TOOLS, *CALENDAR_TOOLS, *LOCATION_TOOLS, *BOOKS_TOOLS, *LANGUAGE_TOOLS, *FOOD_TOOLS, *FUN_TOOLS]
