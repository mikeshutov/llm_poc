from common.config import get_env_bool, get_env_float, get_env_int
from capabilities.adapters.books import search_books
from capabilities.adapters.calendar import get_world_time, public_holidays_lookup
from capabilities.adapters.files import (
    file_create,
    file_delete,
    file_edit,
    file_list,
    file_move,
    file_read,
    get_file_by_id,
    search_asset_details,
    search_assets,
)
from capabilities.adapters.finance import (
    exchange_rates_lookup,
    get_crypto_markets,
    get_stock_price,
)
from capabilities.adapters.food import search_cocktails, search_meals
from capabilities.adapters.games import (
    get_commander_cards,
    get_commander_details,
    get_magic_card_rulings,
    search_magic_cards,
)
from capabilities.adapters.language import define_word
from capabilities.adapters.location import get_caller_location
from capabilities.adapters.math import calculate
from capabilities.adapters.memories import (
    get_memory_detail,
    lookup_evidence,
    search_memories,
)
from capabilities.adapters.news import hn_search
from capabilities.adapters.products import (
    list_product_categories,
    search_products,
)
from capabilities.adapters.profile import (
    set_user_display_name,
    set_user_first_name,
    set_user_last_name,
    update_user_tone,
)
from capabilities.adapters.search import (
    country_lookup,
    news_search,
    structured_facts_lookup,
    wikipedia_search,
)
from capabilities.adapters.web import generic_web_search, web_fetch
from capabilities.adapters.weather import (
    get_current_weather,
    get_historical_month_weather,
)
from capabilities.registry.constants import TOOL_RESULT_TYPE_BOOK_RESULTS
from capabilities.registry.constants import TOOL_RESULT_TYPE_CALCULATION
from capabilities.registry.constants import TOOL_RESULT_TYPE_CALENDAR
from capabilities.registry.constants import TOOL_RESULT_TYPE_CARD_RESULTS
from capabilities.registry.constants import TOOL_RESULT_TYPE_COCKTAIL_RESULTS
from capabilities.registry.constants import TOOL_RESULT_TYPE_COUNTRY
from capabilities.registry.constants import TOOL_RESULT_TYPE_CRYPTO_MARKET
from capabilities.registry.constants import TOOL_RESULT_TYPE_DECKS
from capabilities.registry.constants import TOOL_RESULT_TYPE_DEFINITION
from capabilities.registry.constants import TOOL_RESULT_TYPE_FILE
from capabilities.registry.constants import TOOL_RESULT_TYPE_ASSET_DETAILS
from capabilities.registry.constants import TOOL_RESULT_TYPE_ASSET_RESULTS
from capabilities.registry.constants import TOOL_RESULT_TYPE_FINANCE
from capabilities.registry.constants import TOOL_RESULT_TYPE_GENERIC
from capabilities.registry.constants import TOOL_RESULT_TYPE_KNOWLEDGE
from capabilities.registry.constants import TOOL_RESULT_TYPE_LOCATION
from capabilities.registry.constants import TOOL_RESULT_TYPE_MEAL_RESULTS
from capabilities.registry.constants import TOOL_RESULT_TYPE_MEMORY_DETAIL
from capabilities.registry.constants import TOOL_RESULT_TYPE_MEMORY_RESULTS
from capabilities.registry.constants import TOOL_RESULT_TYPE_NEWS_RESULTS
from capabilities.registry.constants import TOOL_RESULT_TYPE_PRODUCT_CATEGORIES
from capabilities.registry.constants import TOOL_RESULT_TYPE_PRODUCT_RESULTS
from capabilities.registry.constants import TOOL_RESULT_TYPE_PROFILE
from capabilities.registry.constants import TOOL_RESULT_TYPE_RULES
from capabilities.registry.constants import TOOL_RESULT_TYPE_STRUCTURED_FACTS
from capabilities.registry.constants import TOOL_RESULT_TYPE_TIME
from capabilities.registry.constants import TOOL_RESULT_TYPE_TONE
from capabilities.registry.constants import TOOL_RESULT_TYPE_USER_ATTRIBUTE
from capabilities.registry.constants import TOOL_RESULT_TYPE_WEATHER
from common.cache.policy import ToolCachePolicy
from capabilities.retry.policy import RetryPolicy
from capabilities.registry.constants import TOOL_RESULT_TYPE_WEB_SEARCH_RESULTS
from capabilities.registry.constants import TOOL_RESULT_TYPE_WEB_PAGE
from capabilities.registry.models import RateLimitPolicy, Tool, ToolCategory

# Rate limiter and retry policy
BRAVE_RATE_LIMIT_POLICY = RateLimitPolicy(
    max_requests=max(1, get_env_int("BRAVE_RATE_LIMIT_MAX_REQUESTS", 1)),
    window_seconds=max(0.0, get_env_float("BRAVE_RATE_LIMIT_WINDOW_SECONDS", 1.0)),
)
BRAVE_RETRY_POLICY = RetryPolicy(
    max_attempts=max(1, get_env_int("BRAVE_RETRY_MAX_ATTEMPTS", 3)),
    retry_on_timeout=get_env_bool("BRAVE_RETRY_ON_TIMEOUT", True),
    retry_on_429=get_env_bool("BRAVE_RETRY_ON_429", True),
    retry_on_5xx=get_env_bool("BRAVE_RETRY_ON_5XX", True),
    backoff_seconds=max(0.0, get_env_float("BRAVE_RETRY_BACKOFF_SECONDS", 1.0)),
)

# Tool Definitions
PRODUCT_TOOLS = [Tool(search_products, result_type=TOOL_RESULT_TYPE_PRODUCT_RESULTS, cache_policy=ToolCachePolicy(900)), Tool(list_product_categories, result_type=TOOL_RESULT_TYPE_PRODUCT_CATEGORIES, cache_policy=ToolCachePolicy(3600))]
WEATHER_TOOLS = [
    Tool(get_current_weather, result_type=TOOL_RESULT_TYPE_WEATHER, cache_policy=ToolCachePolicy(300)),
    Tool(get_historical_month_weather, result_type=TOOL_RESULT_TYPE_WEATHER, cache_policy=ToolCachePolicy(3600)),
]
FINANCE_TOOLS = [
    Tool(exchange_rates_lookup, result_type=TOOL_RESULT_TYPE_FINANCE, cache_policy=ToolCachePolicy(300)),
    Tool(get_stock_price, result_type=TOOL_RESULT_TYPE_FINANCE, cache_policy=ToolCachePolicy(300)),
]
CRYPTO_TOOLS = [Tool(get_crypto_markets, result_type=TOOL_RESULT_TYPE_CRYPTO_MARKET, cache_policy=ToolCachePolicy(300))]
WEB_SEARCH_TOOLS = [
    Tool(generic_web_search, result_type=TOOL_RESULT_TYPE_WEB_SEARCH_RESULTS, cache_policy=ToolCachePolicy(300), rate_limit_key="brave", retry_policy=BRAVE_RETRY_POLICY, rate_limit_policy=BRAVE_RATE_LIMIT_POLICY),
    Tool(news_search, result_type=TOOL_RESULT_TYPE_NEWS_RESULTS, cache_policy=ToolCachePolicy(300), rate_limit_key="brave", retry_policy=BRAVE_RETRY_POLICY, rate_limit_policy=BRAVE_RATE_LIMIT_POLICY),
    Tool(hn_search, result_type=TOOL_RESULT_TYPE_NEWS_RESULTS, cache_policy=ToolCachePolicy(300)),
    Tool(web_fetch, result_type=TOOL_RESULT_TYPE_WEB_PAGE, cache_policy=ToolCachePolicy(900)),
]
KNOWLEDGE_TOOLS = [
    Tool(wikipedia_search, result_type=TOOL_RESULT_TYPE_KNOWLEDGE, cache_policy=ToolCachePolicy(3600)),
    Tool(structured_facts_lookup, result_type=TOOL_RESULT_TYPE_STRUCTURED_FACTS, cache_policy=ToolCachePolicy(3600)),
    Tool(country_lookup, result_type=TOOL_RESULT_TYPE_COUNTRY, cache_policy=ToolCachePolicy(86400)),
]
CALENDAR_TOOLS = [Tool(public_holidays_lookup, result_type=TOOL_RESULT_TYPE_CALENDAR, cache_policy=ToolCachePolicy(86400)), Tool(get_world_time, result_type=TOOL_RESULT_TYPE_TIME, cache_policy=ToolCachePolicy(30))]
LOCATION_TOOLS = [Tool(get_caller_location, result_type=TOOL_RESULT_TYPE_LOCATION)]
BOOKS_TOOLS = [Tool(search_books, result_type=TOOL_RESULT_TYPE_BOOK_RESULTS, cache_policy=ToolCachePolicy(3600))]
LANGUAGE_TOOLS = [Tool(define_word, result_type=TOOL_RESULT_TYPE_DEFINITION, cache_policy=ToolCachePolicy(86400))]
FOOD_TOOLS = [
    Tool(
        search_meals,
        result_type=TOOL_RESULT_TYPE_MEAL_RESULTS,
        cache_policy=ToolCachePolicy(3600),
    ),
    Tool(search_cocktails, result_type=TOOL_RESULT_TYPE_COCKTAIL_RESULTS, cache_policy=ToolCachePolicy(3600)),
]
MATH_TOOLS = [Tool(calculate, result_type=TOOL_RESULT_TYPE_CALCULATION, cache_policy=ToolCachePolicy(86400))]
GAMES_TOOLS = [
    Tool(get_commander_details, result_type=TOOL_RESULT_TYPE_DECKS, cache_policy=ToolCachePolicy(3600)),
    Tool(get_commander_cards, result_type=TOOL_RESULT_TYPE_CARD_RESULTS, cache_policy=ToolCachePolicy(3600)),
    Tool(search_magic_cards, result_type=TOOL_RESULT_TYPE_CARD_RESULTS, cache_policy=ToolCachePolicy(3600)),
    Tool(get_magic_card_rulings, result_type=TOOL_RESULT_TYPE_RULES, cache_policy=ToolCachePolicy(3600)),
]
MEMORY_TOOLS = [
    Tool(get_memory_detail, result_type=TOOL_RESULT_TYPE_MEMORY_DETAIL, cache_policy=ToolCachePolicy(300, is_user_specific=True)),
    Tool(lookup_evidence, result_type=TOOL_RESULT_TYPE_MEMORY_RESULTS, cache_policy=ToolCachePolicy(300, is_user_specific=True)),
    Tool(search_memories, result_type=TOOL_RESULT_TYPE_MEMORY_RESULTS, cache_policy=ToolCachePolicy(300, is_user_specific=True)),
]
FILE_TOOLS = [Tool(search_assets, result_type=TOOL_RESULT_TYPE_ASSET_RESULTS, cache_policy=ToolCachePolicy(300, is_user_specific=True)), Tool(search_asset_details, result_type=TOOL_RESULT_TYPE_ASSET_DETAILS, cache_policy=ToolCachePolicy(300, is_user_specific=True)), Tool(get_file_by_id, result_type=TOOL_RESULT_TYPE_FILE, cache_policy=ToolCachePolicy(300, is_user_specific=True))]
FILE_MANAGEMENT_TOOLS = [
    Tool(file_create, result_type=TOOL_RESULT_TYPE_FILE),
    Tool(file_read, result_type=TOOL_RESULT_TYPE_FILE),
    Tool(file_edit, result_type=TOOL_RESULT_TYPE_FILE),
    Tool(file_delete, result_type=TOOL_RESULT_TYPE_FILE),
    Tool(file_list, result_type=TOOL_RESULT_TYPE_FILE),
    Tool(file_move, result_type=TOOL_RESULT_TYPE_FILE),
]
PROFILE_TOOLS = [
    Tool(
        set_user_display_name,
        result_type=TOOL_RESULT_TYPE_PROFILE,
        rules=(
            "Use when the user gives a nickname, preferred short name, or says phrasing such as 'call me ...'.",
            "Do not use for a legal or explicitly identified first or last name.",
        ),
    ),
    Tool(
        set_user_first_name,
        result_type=TOOL_RESULT_TYPE_PROFILE,
        rules=(
            "Use only when first_name is currently missing and the user explicitly provides their first name in this turn.",
            "Do not use for nicknames, preferred short names, guesses, or paraphrases.",
        ),
    ),
    Tool(
        set_user_last_name,
        result_type=TOOL_RESULT_TYPE_PROFILE,
        rules=(
            "Use only when last_name is currently missing and the user explicitly provides their last name in this turn.",
            "Do not use for nicknames, preferred short names, guesses, or paraphrases.",
        ),
    ),
    Tool(
        update_user_tone,
        result_type=TOOL_RESULT_TYPE_TONE,
        rules=(
            "Use only when the user clearly expresses an ongoing preference for how responses should sound or be structured.",
            "Pass confidence from 0 to 1; values below 0.9 are rejected.",
            "Pass only explicitly supported fields, preserve other stored fields, and do not call when requested values are already stored.",
        ),
    ),
]

# if this were to grow much larger I would probably create sub categories or a tree structure of tools
TOOL_CATEGORIES: dict[str, ToolCategory] = {
    "products": ToolCategory(
        tools=PRODUCT_TOOLS,
        description="Search and browse products and product categories from the internal catalog.",
        rules=[
            "Make sure that previous context is taken into account when providing filters unless explicitly told not to.",
        ]
    ),
    "weather": ToolCategory(
        tools=WEATHER_TOOLS,
        description="Retrieve current or historical weather conditions for a specified location.",
        rules=[
            "Use weather_current for current weather, temperature, or conditions requests.",
            "Use weather_historical_month only when the user specifies a past month or historical period.",
            "Do not answer current-weather requests from conversation summaries or older weather results.",
        ],
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
        description="Search the web, current news, or Hacker News discussions for fresh external results.",
        rules=[
            "Use web_fetch when the user provides a URL or asks to inspect, summarize, or extract details from a specific page.",
            "Use search_news for current, recent, latest, breaking, or location-based news requests.",
            "Use search_hacker_news for Hacker News stories, comments, and discussions.",
            "Use web_search for non-news web information or broader web research.",
            "Use a live search for fresh information; do not rely on conversation summaries or older search results unless the user explicitly asks about them.",
            "Use prior search results and fetched page evidence to refine follow-up searches when needed.",
        ],
    ),
    "knowledge": ToolCategory(
        tools=KNOWLEDGE_TOOLS,
        description="Look up information from Wikipedia, Wikidata structured facts, or country data.",
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
    "math": ToolCategory(
        tools=MATH_TOOLS,
        description="Evaluate mathematical expressions and perform mathematical calculations.",
    ),
    "games": ToolCategory(
        tools=GAMES_TOOLS,
        description="Look up Magic: The Gathering commander deck context from EDHREC and card information from Scryfall, including reranked commander card recommendations, oracle text, mana cost, type line, legality, pricing, and images.",
    ),
    "memories": ToolCategory(
        tools=MEMORY_TOOLS,
        description="Search prior conversation summaries for relevant past requests and discussions as memories.",
        rules=[
            "Use memory_search without conversation_ids to discover relevant conversations, then use it again with returned conversation_ids to find specific historical mentions or exchanges.",
            "Use memory_detail after memory_search when you need the exact prior prompt, response, or structured payload for one memory hit.",
            "Use memory_lookup_evidence with an array of evidence IDs from prior conversation messages when you need the full records behind prior cited evidence.",
            "When the user asks what was previously said, decided, suggested, or discussed about a topic, prefer the two-step memories flow over guessing from current context.",
        ],
    ),
    "files": ToolCategory(
        tools=FILE_TOOLS,
        description="To be utilized for any searches involving files. Search and retrieve content from uploaded files. To be used when files are in the context either with a name or ID.",
        result_rules=[
            "If a file_id is already known in the evidence or context, treat it as the authoritative file reference rather than implying a new file discovery step.",
            "When an asset_id is known, prefer conclusions grounded in file_get or asset_search_details results over vague name inference.",
            "Summarize or extract relevant pieces unless a quote is more appropriate.",
            "When referencing file content, cite the file name.",
            "Cite the relevant file evidence IDs. The renderer displays cited file names inline; do not invent URLs, Markdown links, or file paths.",
            "When comparing an image from a file, load its description first and use the description rather than the file name.",
        ],
    ),
    "file_management": ToolCategory(
        tools=FILE_MANAGEMENT_TOOLS,
        description="Create, read, edit, delete, list, rename, and move files in the managed file workspace.",
        rules=[
            "Use only for explicit file-management requests; do not use to answer ordinary file-search questions.",
            "Never invent a file_id. Use file_list or existing context first when the target is ambiguous.",
            "Keep all paths inside the managed files directory.",
        ],
    ),
}

tools = [*PRODUCT_TOOLS, *WEATHER_TOOLS, *FINANCE_TOOLS, *CRYPTO_TOOLS, *WEB_SEARCH_TOOLS, *KNOWLEDGE_TOOLS, *CALENDAR_TOOLS, *LOCATION_TOOLS, *BOOKS_TOOLS, *LANGUAGE_TOOLS, *FOOD_TOOLS, *MATH_TOOLS, *GAMES_TOOLS, *MEMORY_TOOLS, *FILE_TOOLS, *FILE_MANAGEMENT_TOOLS, *PROFILE_TOOLS]

TOOLS_BY_NAME = {tool.name: tool for tool in tools}


def get_tool_result_type(name: str) -> str:
    tool = TOOLS_BY_NAME.get(name)
    if tool is None:
        return TOOL_RESULT_TYPE_GENERIC
    return tool.result_type or TOOL_RESULT_TYPE_GENERIC

