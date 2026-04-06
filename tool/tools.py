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


@dataclass
class ToolCategory:
    tools: list
    description: str
    rules: list[str] = field(default_factory=list)
    result_rules: list[str] = field(default_factory=list)


PRODUCT_TOOLS = [find_products, list_product_categories]
WEATHER_TOOLS = [resolve_city_location, get_current_weather, get_historical_month_weather]
FINANCE_TOOLS = [exchange_rates_lookup, exchange_rates_time_series, get_latest_exchange_rates, get_stock_price]
CRYPTO_TOOLS = [get_crypto_markets]
SEARCH_TOOLS = [generic_web_search, news_search, wikipedia_search, structured_facts_lookup, hn_search, country_lookup]
CALENDAR_TOOLS = [public_holidays_lookup, get_world_time]
LOCATION_TOOLS = [get_caller_location]
BOOKS_TOOLS = [search_books]
LANGUAGE_TOOLS = [define_word]
FOOD_TOOLS = [search_meals, search_cocktails]
FUN_TOOLS = [get_advice, get_quote, get_astronomy_picture]
MATH_TOOLS = [calculate]
FILE_TOOLS = [search_files, search_file_for_details]

# if this were to grow much larger I would probably create sub categories or a tree structure of tools
TOOL_CATEGORIES: dict[str, ToolCategory] = {
    "products": ToolCategory(
        tools=PRODUCT_TOOLS,
        description="Search and browse products and product categories from the catalog.",
        rules=[
            "For product searches utilize internal tools first before web searches."
            "Make sure that previous context is taken into account when providing filters unless explicitely told not to."],
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
    "search": ToolCategory(
        tools=SEARCH_TOOLS,
        description="Search the web, news, Wikipedia, structured knowledge, or country information for general information about any topic.",
        rules=["If you use Brave/WebSearch tools, use at most ONE of them in the entire plan.","You can use the Wiki tools multiple times as needed."],
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
        description="Search and retrieve content from uploaded files. To be used when files are in the context either with a name or ID.",
        rules=[
            "Use search_files to discover files and obtain their file_id.",
            "Use search_file_for_details with the file_id and a specific query to retrieve details from a file.",
            "Always call search_file_for_details over relying on context when the question is concerning files.",
        ],
        result_rules=[
            "Summarize or extract relevant pieces unless a quote is more appropriate.",
            "When referencing file content, cite the file name and add a clickable link.",
            "When posting a list of files put them in a bullet list with links to the file.",
            "Make sure to use markdown. Also ensure that spaces in file names are converted to %20.",
            "When evidence contains a file_path or name for an image (jpg, jpeg, png, webp), render it using markdown image syntax: ![file_name](/app/static/files/file_name).",
            "When evidence contains a file_path or name for a document (pdf, txt, docx), render it as a markdown link: [file_name](/app/static/files/file_name).",
        ],
    ),
}

tools = [*PRODUCT_TOOLS, *WEATHER_TOOLS, *FINANCE_TOOLS, *CRYPTO_TOOLS, *SEARCH_TOOLS, *CALENDAR_TOOLS, *LOCATION_TOOLS, *BOOKS_TOOLS, *LANGUAGE_TOOLS, *FOOD_TOOLS, *FUN_TOOLS, *MATH_TOOLS, *FILE_TOOLS]
