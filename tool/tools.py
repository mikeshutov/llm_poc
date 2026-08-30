from common.config import get_env_bool, get_env_float, get_env_int
from request_orchestrator.shared.tool_adapter.books import search_books
from request_orchestrator.shared.tool_adapter.calendar import get_world_time, public_holidays_lookup
from request_orchestrator.shared.tool_adapter.files import get_file_by_id, search_file_for_details, search_files
from request_orchestrator.shared.tool_adapter.finance import (
    exchange_rates_lookup,
    exchange_rates_time_series,
    get_crypto_markets,
    get_latest_exchange_rates,
    get_stock_price,
)
from request_orchestrator.shared.tool_adapter.food import search_cocktails, search_meals
from request_orchestrator.shared.tool_adapter.fun import get_advice, get_astronomy_picture, get_quote
from request_orchestrator.shared.tool_adapter.games import (
    get_commander_cards,
    get_commander_details,
    get_magic_card_rulings,
    search_magic_cards,
)
from request_orchestrator.shared.tool_adapter.language import define_word
from request_orchestrator.shared.tool_adapter.location import get_caller_location
from request_orchestrator.shared.tool_adapter.math import calculate
from request_orchestrator.shared.tool_adapter.memories import (
    get_memory_detail,
    lookup_evidence,
    search_memories,
    search_roundtrip_memories,
)
from request_orchestrator.shared.tool_adapter.news import hn_search
from request_orchestrator.shared.tool_adapter.products import (
    find_products,
    find_products_web,
    list_product_categories,
)
from request_orchestrator.shared.tool_adapter.profile import (
    set_user_display_name,
    set_user_first_name,
    set_user_last_name,
    update_user_tone,
)
from request_orchestrator.shared.tool_adapter.search import (
    country_lookup,
    generic_web_search,
    news_search,
    structured_facts_lookup,
    wikipedia_search,
)
from request_orchestrator.shared.tool_adapter.user_attributes import (
    create_user_attribute,
    get_user_attributes,
    search_user_attributes,
    update_user_attribute,
)
from request_orchestrator.shared.tool_adapter.weather import (
    get_current_weather,
    get_historical_month_weather,
    resolve_city_location,
)
from tool.constants import TOOL_RESULT_TYPE_ADVICE
from tool.constants import TOOL_RESULT_TYPE_ASTRONOMY_PICTURE
from tool.constants import TOOL_RESULT_TYPE_BOOK_RESULTS
from tool.constants import TOOL_RESULT_TYPE_CALCULATION
from tool.constants import TOOL_RESULT_TYPE_CALENDAR
from tool.constants import TOOL_RESULT_TYPE_CARD_RESULTS
from tool.constants import TOOL_RESULT_TYPE_COCKTAIL_RESULTS
from tool.constants import TOOL_RESULT_TYPE_COUNTRY
from tool.constants import TOOL_RESULT_TYPE_CRYPTO_MARKET
from tool.constants import TOOL_RESULT_TYPE_DECKS
from tool.constants import TOOL_RESULT_TYPE_DEFINITION
from tool.constants import TOOL_RESULT_TYPE_FILE
from tool.constants import TOOL_RESULT_TYPE_FILE_DETAILS
from tool.constants import TOOL_RESULT_TYPE_FILE_RESULTS
from tool.constants import TOOL_RESULT_TYPE_FINANCE
from tool.constants import TOOL_RESULT_TYPE_GENERIC
from tool.constants import TOOL_RESULT_TYPE_KNOWLEDGE
from tool.constants import TOOL_RESULT_TYPE_LOCATION
from tool.constants import TOOL_RESULT_TYPE_MEAL_RESULTS
from tool.constants import TOOL_RESULT_TYPE_MEMORY_DETAIL
from tool.constants import TOOL_RESULT_TYPE_MEMORY_RESULTS
from tool.constants import TOOL_RESULT_TYPE_NEWS_RESULTS
from tool.constants import TOOL_RESULT_TYPE_PRODUCT_CATEGORIES
from tool.constants import TOOL_RESULT_TYPE_PRODUCT_RESULTS
from tool.constants import TOOL_RESULT_TYPE_PROFILE
from tool.constants import TOOL_RESULT_TYPE_QUOTE
from tool.constants import TOOL_RESULT_TYPE_RULES
from tool.constants import TOOL_RESULT_TYPE_STRUCTURED_FACTS
from tool.constants import TOOL_RESULT_TYPE_TIME
from tool.constants import TOOL_RESULT_TYPE_TONE
from tool.constants import TOOL_RESULT_TYPE_USER_ATTRIBUTE
from tool.constants import TOOL_RESULT_TYPE_WEATHER
from tool.constants import TOOL_RESULT_TYPE_WEB_SEARCH_RESULTS
from tool.models import RateLimitPolicy, RetryPolicy, Tool, ToolCategory

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
PRODUCT_TOOLS = [Tool(find_products, result_type=TOOL_RESULT_TYPE_PRODUCT_RESULTS), Tool(list_product_categories, result_type=TOOL_RESULT_TYPE_PRODUCT_CATEGORIES)]
PRODUCT_WEB_TOOLS = [Tool(find_products_web, result_type=TOOL_RESULT_TYPE_PRODUCT_RESULTS)]
WEATHER_TOOLS = [
    Tool(resolve_city_location, result_type=TOOL_RESULT_TYPE_LOCATION),
    Tool(get_current_weather, result_type=TOOL_RESULT_TYPE_WEATHER),
    Tool(get_historical_month_weather, result_type=TOOL_RESULT_TYPE_WEATHER),
]
FINANCE_TOOLS = [
    Tool(exchange_rates_lookup, result_type=TOOL_RESULT_TYPE_FINANCE),
    Tool(exchange_rates_time_series, result_type=TOOL_RESULT_TYPE_FINANCE),
    Tool(get_latest_exchange_rates, result_type=TOOL_RESULT_TYPE_FINANCE),
    Tool(get_stock_price, result_type=TOOL_RESULT_TYPE_FINANCE),
]
CRYPTO_TOOLS = [Tool(get_crypto_markets, result_type=TOOL_RESULT_TYPE_CRYPTO_MARKET)]
WEB_SEARCH_TOOLS = [
    Tool(generic_web_search, result_type=TOOL_RESULT_TYPE_WEB_SEARCH_RESULTS, rate_limit_key="brave", retry_policy=BRAVE_RETRY_POLICY, rate_limit_policy=BRAVE_RATE_LIMIT_POLICY),
    Tool(news_search, result_type=TOOL_RESULT_TYPE_NEWS_RESULTS, rate_limit_key="brave", retry_policy=BRAVE_RETRY_POLICY, rate_limit_policy=BRAVE_RATE_LIMIT_POLICY),
]
KNOWLEDGE_TOOLS = [
    Tool(wikipedia_search, result_type=TOOL_RESULT_TYPE_KNOWLEDGE),
    Tool(structured_facts_lookup, result_type=TOOL_RESULT_TYPE_STRUCTURED_FACTS),
    Tool(hn_search, result_type=TOOL_RESULT_TYPE_NEWS_RESULTS),
    Tool(country_lookup, result_type=TOOL_RESULT_TYPE_COUNTRY),
]
CALENDAR_TOOLS = [Tool(public_holidays_lookup, result_type=TOOL_RESULT_TYPE_CALENDAR), Tool(get_world_time, result_type=TOOL_RESULT_TYPE_TIME)]
LOCATION_TOOLS = [Tool(get_caller_location, result_type=TOOL_RESULT_TYPE_LOCATION)]
BOOKS_TOOLS = [Tool(search_books, result_type=TOOL_RESULT_TYPE_BOOK_RESULTS)]
LANGUAGE_TOOLS = [Tool(define_word, result_type=TOOL_RESULT_TYPE_DEFINITION)]
FOOD_TOOLS = [Tool(search_meals, result_type=TOOL_RESULT_TYPE_MEAL_RESULTS), Tool(search_cocktails, result_type=TOOL_RESULT_TYPE_COCKTAIL_RESULTS)]
FUN_TOOLS = [Tool(get_advice, result_type=TOOL_RESULT_TYPE_ADVICE), Tool(get_quote, result_type=TOOL_RESULT_TYPE_QUOTE), Tool(get_astronomy_picture, result_type=TOOL_RESULT_TYPE_ASTRONOMY_PICTURE)]
MATH_TOOLS = [Tool(calculate, result_type=TOOL_RESULT_TYPE_CALCULATION)]
GAMES_TOOLS = [
    Tool(get_commander_details, result_type=TOOL_RESULT_TYPE_DECKS),
    Tool(get_commander_cards, result_type=TOOL_RESULT_TYPE_CARD_RESULTS),
    Tool(search_magic_cards, result_type=TOOL_RESULT_TYPE_CARD_RESULTS),
    Tool(get_magic_card_rulings, result_type=TOOL_RESULT_TYPE_RULES),
]
MEMORY_TOOLS = [
    Tool(get_memory_detail, result_type=TOOL_RESULT_TYPE_MEMORY_DETAIL),
    Tool(lookup_evidence, result_type=TOOL_RESULT_TYPE_MEMORY_RESULTS),
    Tool(search_memories, result_type=TOOL_RESULT_TYPE_MEMORY_RESULTS),
    Tool(search_roundtrip_memories, result_type=TOOL_RESULT_TYPE_MEMORY_RESULTS),
]
USER_ATTRIBUTE_TOOLS = [Tool(create_user_attribute, result_type=TOOL_RESULT_TYPE_USER_ATTRIBUTE), Tool(update_user_attribute, result_type=TOOL_RESULT_TYPE_USER_ATTRIBUTE), Tool(get_user_attributes, result_type=TOOL_RESULT_TYPE_USER_ATTRIBUTE), Tool(search_user_attributes, result_type=TOOL_RESULT_TYPE_USER_ATTRIBUTE)]
FILE_TOOLS = [Tool(search_files, result_type=TOOL_RESULT_TYPE_FILE_RESULTS), Tool(search_file_for_details, result_type=TOOL_RESULT_TYPE_FILE_DETAILS), Tool(get_file_by_id, result_type=TOOL_RESULT_TYPE_FILE)]
PROFILE_TOOLS = [Tool(set_user_display_name, result_type=TOOL_RESULT_TYPE_PROFILE), Tool(set_user_first_name, result_type=TOOL_RESULT_TYPE_PROFILE), Tool(set_user_last_name, result_type=TOOL_RESULT_TYPE_PROFILE), Tool(update_user_tone, result_type=TOOL_RESULT_TYPE_TONE)]

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
    "games": ToolCategory(
        tools=GAMES_TOOLS,
        description="Look up Magic: The Gathering commander deck context from EDHREC and card information from Scryfall, including reranked commander card recommendations, oracle text, mana cost, type line, legality, pricing, and images.",
    ),
    "memories": ToolCategory(
        tools=MEMORY_TOOLS,
        description="Search prior conversation summaries for relevant past requests and discussions as memories.",
        rules=[
            "Use search_memories first to locate the most relevant conversations for a topic or prior discussion.",
            "Use search_roundtrip_memories after search_memories when you need specific historical mentions or exchanges inside those conversations.",
            "Use get_memory_detail after search_roundtrip_memories when you need the exact prior prompt, response, or structured payload for one memory hit.",
            "Use lookup_evidence with an array of evidence IDs from recent_roundtrips when you need the full records behind prior cited evidence.",
            "When the user asks what was previously said, decided, suggested, or discussed about a topic, prefer the two-step memories flow over guessing from current context.",
        ],
    ),
    "user_attributes": ToolCategory(
        tools=USER_ATTRIBUTE_TOOLS,
        description="Create, update, list, and search persistent user attributes using typed categories like career.likes, projects.goals, or food.dislikes. Attribute values are stored as arrays/lists of strings.",
        rules=[
            "Use user_attributes when the task is about stable user-specific facts, preferences, interests, likes, dislikes, favorites, skills, goals, or something the assistant should explicitly retain beyond the current conversation.",
            "Some relevant user attributes may already be pre-loaded into the user_profile.user_attributes section of the prompt for this request. Do not call get_user_attributes or search_user_attributes just to re-fetch data that is already visible there.",
            "Use create_user_attribute to store a new user attribute, update_user_attribute to revise or deactivate one, get_user_attributes only when you need a broader or differently filtered view than what is already in user_profile.user_attributes, and search_user_attributes when you need targeted retrieval beyond the attributes already shown in the profile. For search_user_attributes, keep the query short, literal, and attribute-focused rather than a long paraphrase or inferred summary.",
            "When a potential new attribute may overlap with, refine, replace, or merge into existing profile data, fetch relevant existing attributes first so update/refactor work is prioritized over blind creation.",
            "Do not treat a lightly refactored phrasing of the same underlying term as a required update. If the core term or concept is already present, only update when the new phrasing adds materially new information rather than just restating it differently.",
            "Prefer conversation memories for prior discussion recall, and prefer user_attributes for durable user profile characteristics. Prefer the already-provided user profile before planning extra user_attributes reads.",
            "A user interest, preference, like, dislike, favorite, skill, goal, or recurring characteristic should generally be modeled as a user attribute. Use the *.goals qualifier for durable aims such as career objectives, project objectives, or fitness targets. When creating or updating one, the value field should be a JSON array/list of strings. Store concrete user-specific entries only, not category labels, summaries, placeholders, or brace-wrapped descriptions like `{'dietary staples mentioned by the user'}`."
        ],
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
            "If a file_id is already known in the evidence or context, treat it as the authoritative file reference rather than implying a new file discovery step.",
            "When a file_id is known, prefer conclusions grounded in get_file_by_id or search_file_for_details results over vague file-name inference.",
            "Summarize or extract relevant pieces unless a quote is more appropriate.",
            "When referencing file content, cite the file name.",
            "When rendering a list of files put them in a markdown list with links to the files.",
        ],
    ),
}

tools = [*PRODUCT_TOOLS, *PRODUCT_WEB_TOOLS, *WEATHER_TOOLS, *FINANCE_TOOLS, *CRYPTO_TOOLS, *WEB_SEARCH_TOOLS, *KNOWLEDGE_TOOLS, *CALENDAR_TOOLS, *LOCATION_TOOLS, *BOOKS_TOOLS, *LANGUAGE_TOOLS, *FOOD_TOOLS, *FUN_TOOLS, *MATH_TOOLS, *GAMES_TOOLS, *MEMORY_TOOLS, *USER_ATTRIBUTE_TOOLS, *FILE_TOOLS, *PROFILE_TOOLS]

TOOLS_BY_NAME = {tool.name: tool for tool in tools}


def get_tool_result_type(name: str) -> str:
    tool = TOOLS_BY_NAME.get(name)
    if tool is None:
        return TOOL_RESULT_TYPE_GENERIC
    return tool.result_type or TOOL_RESULT_TYPE_GENERIC

