from dataclasses import dataclass, field

from agent.tool_adapter.products.find_products import find_products
from agent.tool_adapter.products.find_products_web import find_products_web
from agent.tool_adapter.products.list_product_categories import list_product_categories
from agent.tool_adapter.weather.get_current_weather import get_current_weather
from agent.tool_adapter.weather.get_historical_month_weather import get_historical_month_weather
from agent.tool_adapter.weather.resolve_city_location import resolve_city_location
from agent.tool_adapter.finance.exchange_rates_lookup import exchange_rates_lookup
from agent.tool_adapter.finance.exchange_rates_time_series import exchange_rates_time_series
from agent.tool_adapter.finance.crypto_markets import get_crypto_markets
from agent.tool_adapter.finance.latest_exchange_rates import get_latest_exchange_rates
from agent.tool_adapter.finance.get_stock_price import get_stock_price
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
from agent.tool_adapter.math.calculate import calculate
from agent.tool_adapter.search.country_lookup import country_lookup
from agent.tool_adapter.calendar.world_time import get_world_time
from agent.tool_adapter.files.search_file_for_details import search_file_for_details
from agent.tool_adapter.files.search_files import search_files
from agent.tool_adapter.files.get_file_by_id import get_file_by_id


@dataclass
class ToolCategory:
    tools: list
    description: str
    rules: list[str] = field(default_factory=list)
    result_rules: list[str] = field(default_factory=list)


PRODUCT_TOOLS = [find_products, list_product_categories]
PRODUCT_WEB_TOOLS = [find_products_web]
WEATHER_TOOLS = [resolve_city_location, get_current_weather, get_historical_month_weather]
FINANCE_TOOLS = [exchange_rates_lookup, exchange_rates_time_series, get_latest_exchange_rates, get_stock_price]
CRYPTO_TOOLS = [get_crypto_markets]
WEB_SEARCH_TOOLS = [generic_web_search, news_search]
KNOWLEDGE_TOOLS = [wikipedia_search, structured_facts_lookup, hn_search, country_lookup]
CALENDAR_TOOLS = [public_holidays_lookup, get_world_time]
LOCATION_TOOLS = [get_caller_location]
BOOKS_TOOLS = [search_books]
LANGUAGE_TOOLS = [define_word]
FOOD_TOOLS = [search_meals, search_cocktails]
FUN_TOOLS = [get_advice, get_quote, get_astronomy_picture]
MATH_TOOLS = [calculate]
FILE_TOOLS = [search_files, search_file_for_details, get_file_by_id]

# if this were to grow much larger I would probably create sub categories or a tree structure of tools
TOOL_CATEGORIES: dict[str, ToolCategory] = {
    "products": ToolCategory(
        tools=PRODUCT_TOOLS,
        description="Search and browse products and product categories from the internal catalog.",
        rules=[
            "Make sure that previous context is taken into account when providing filters unless explicitly told not to.",
            "When utilizing an image for comparison make sure that we load its description first. Utilize the description not the file name.",
        ]
    ),
    "products_web": ToolCategory(
        tools=PRODUCT_WEB_TOOLS,
        description="Search the web for products when the internal catalog returns no results.",
        rules=["Only use when the internal catalog has no results."],
    ),
    "weather": ToolCategory(
        tools=WEATHER_TOOLS,
        description="Look up current or historical weather conditions for a city.",
    ),
    "finance": ToolCategory(
        tools=FINANCE_TOOLS,
        description="Retrieve currency exchange rates, historical rate time series, stock prices, and commodity prices (e.g. gold, silver via futures tickers like GC=F, SI=F).",
    ),
    "finance_crypto": ToolCategory(
        tools=CRYPTO_TOOLS,
        description="Retrieve live cryptocurrency market data including prices, market cap, and volume.",
    ),
    "web_search": ToolCategory(
        tools=WEB_SEARCH_TOOLS,
        description="Search the web or news using Brave for general information about any topic.",
        rules=["Use at most ONE web search tool call in the entire plan."],
    ),
    "knowledge": ToolCategory(
        tools=KNOWLEDGE_TOOLS,
        description="Look up information from Wikipedia, Wikidata structured facts, Hacker News, or country data.",
        rules=["These tools can be used multiple times as needed."],
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
    "math": ToolCategory(
        tools=MATH_TOOLS,
        description="Evaluate mathematical expressions and perform mathematical calculations.",
    ),
    "files": ToolCategory(
        tools=FILE_TOOLS,
        description="To be utilized for any searches involving files. Search and retrieve content from uploaded files. To be used when files are in the context either with a name or ID.",
        rules=[
            "When a file_id is present in the context, always call get_file_by_id first before using file content as input to any other tool. Never infer or guess file content from the file name alone.",
            "Use search_files to discover files and obtain their file_id when no file_id is in context.",
            "Use get_file_by_id with a known file_id to retrieve a preview of the file contents.",
            "Use search_file_for_details with the file_id and a specific query to retrieve deeper details from a file.",
        ],
        result_rules=[
            "Summarize or extract relevant pieces unless a quote is more appropriate.",
            "When referencing file content, cite the file name.",
            "When rendering a list of files put them in a markdown list with links to the files.",
        ],
    ),
}

tools = [*PRODUCT_TOOLS, *PRODUCT_WEB_TOOLS, *WEATHER_TOOLS, *FINANCE_TOOLS, *CRYPTO_TOOLS, *WEB_SEARCH_TOOLS, *KNOWLEDGE_TOOLS, *CALENDAR_TOOLS, *LOCATION_TOOLS, *BOOKS_TOOLS, *LANGUAGE_TOOLS, *FOOD_TOOLS, *FUN_TOOLS, *MATH_TOOLS, *FILE_TOOLS]
