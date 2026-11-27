from __future__ import annotations

from dataclasses import dataclass
from capabilities.registry.constants import (
    TOOL_CATEGORY_BOOKS,
    TOOL_CATEGORY_CALENDAR,
    TOOL_CATEGORY_FINANCE,
    TOOL_CATEGORY_KNOWLEDGE,
    TOOL_CATEGORY_LANGUAGE,
    TOOL_CATEGORY_LOCATION,
    TOOL_CATEGORY_MATH,
    TOOL_CATEGORY_PRODUCTS,
    TOOL_CATEGORY_WEATHER,
    TOOL_CATEGORY_WORKSPACE,
    TOOL_CATEGORY_WRITING,
    TOOL_DISCOVERY_CATEGORY_ASSETS,
    TOOL_DISCOVERY_CATEGORY_CODE_EXECUTION,
    TOOL_DISCOVERY_CATEGORY_FILE_MANAGEMENT,
    TOOL_DISCOVERY_CATEGORY_INTERNAL_PRODUCTS,
    TOOL_DISCOVERY_CATEGORY_MAGIC,
    TOOL_DISCOVERY_CATEGORY_MEMORIES,
    TOOL_DISCOVERY_CATEGORY_NEWS,
    TOOL_DISCOVERY_CATEGORY_RECIPES,
    TOOL_DISCOVERY_CATEGORY_UPLOADED_FILES,
    TOOL_DISCOVERY_CATEGORY_USER_PROFILE,
    TOOL_DISCOVERY_CATEGORY_WEB_SEARCH,
)


@dataclass(frozen=True)
class ToolDiscoveryMetadata:
    title: str
    category: str
    description: str
    scope_terms: tuple[str, ...] = ()
    version: int = 3

    def embedding_text(self, tool_name: str) -> str:
        parts = [tool_name, self.title, self.category, self.description]
        if self.scope_terms:
            parts.append("Scope: " + ", ".join(self.scope_terms))
        return "\n".join(parts)


DISCOVERY_METADATA: dict[str, ToolDiscoveryMetadata] = {
    "products_search": ToolDiscoveryMetadata("Product search", TOOL_CATEGORY_PRODUCTS, "Search internal catalog products, external shopping results, or both using an explicit source.", ("catalog", "inventory", "web shopping", "price")),
    "products_list_categories": ToolDiscoveryMetadata("Internal product category lookup", TOOL_DISCOVERY_CATEGORY_INTERNAL_PRODUCTS, "List categories available in the internal product catalog.", ("catalog", "categories")),
    "weather_lookup": ToolDiscoveryMetadata("Weather lookup", TOOL_CATEGORY_WEATHER, "Retrieve current, forecast, or historical monthly weather conditions for a city or location.", ("temperature", "wind", "conditions", "forecast", "historical weather")),
    "finance_exchange_rates": ToolDiscoveryMetadata("Currency exchange rates", TOOL_CATEGORY_FINANCE, "Retrieve latest, historical-date, or date-range exchange rates between currencies."),
    "finance_stock_price": ToolDiscoveryMetadata("Stock market price lookup", TOOL_CATEGORY_FINANCE, "Retrieve current stock prices and basic market data by ticker."),
    "finance_crypto_markets": ToolDiscoveryMetadata("Cryptocurrency market lookup", TOOL_CATEGORY_FINANCE, "Retrieve cryptocurrency prices, market caps, volume, and market changes."),
    "finance_crypto_historical_prices": ToolDiscoveryMetadata("Historical cryptocurrency prices", TOOL_CATEGORY_FINANCE, "Retrieve historical cryptocurrency prices for an inclusive date range."),
    "web_search": ToolDiscoveryMetadata("General web search", TOOL_DISCOVERY_CATEGORY_WEB_SEARCH, "Search the public web for general information and current web content.", ("internet", "web pages")),
    "web_fetch": ToolDiscoveryMetadata("Web page fetch", TOOL_DISCOVERY_CATEGORY_WEB_SEARCH, "Fetch a public web page and extract its readable title, description, and text for detailed research.", ("internet", "web pages", "article details")),
    "search_news": ToolDiscoveryMetadata("Current news search", TOOL_DISCOVERY_CATEGORY_NEWS, "Search current news stories and article overviews.", ("breaking news", "recent events")),
    "search_wikipedia": ToolDiscoveryMetadata("Wikipedia search", TOOL_CATEGORY_KNOWLEDGE, "Search Wikipedia pages and retrieve short article summaries."),
    "search_facts": ToolDiscoveryMetadata("Structured knowledge lookup", TOOL_CATEGORY_KNOWLEDGE, "Retrieve stable entity and relationship facts from Wikidata using structured queries.", ("SPARQL", "facts", "entities")),
    "search_hacker_news": ToolDiscoveryMetadata("Hacker News search", TOOL_DISCOVERY_CATEGORY_NEWS, "Search Hacker News stories, comments, and discussions."),
    "search_country": ToolDiscoveryMetadata("Country information lookup", TOOL_CATEGORY_KNOWLEDGE, "Retrieve country details such as capital, region, population, currencies, and languages."),
    "calendar_public_holidays": ToolDiscoveryMetadata("Public holiday lookup", TOOL_CATEGORY_CALENDAR, "Retrieve official public holidays for a country and year, optionally filtered to a province or state.", ("province", "state", "subdivision")),
    "calendar_world_time": ToolDiscoveryMetadata("World time lookup", TOOL_CATEGORY_CALENDAR, "Retrieve the current local date and time for an IANA timezone."),
    "location_caller": ToolDiscoveryMetadata("Caller location lookup", TOOL_CATEGORY_LOCATION, "Retrieve the caller's approximate country, region, city, timezone, and coordinates from IP location."),
    "books_search": ToolDiscoveryMetadata("Book catalog search", TOOL_CATEGORY_BOOKS, "Search the Open Library catalog by book title, author, subject, or keyword."),
    "language_define": ToolDiscoveryMetadata("Word definition lookup", TOOL_CATEGORY_LANGUAGE, "Retrieve definitions, parts of speech, examples, synonyms, and antonyms for an English word."),
    "food_recipes": ToolDiscoveryMetadata("Recipe search", TOOL_DISCOVERY_CATEGORY_RECIPES, "Search recipes by name or keyword, including ingredients, categories, cuisine areas, and cooking instructions."),
    "food_cocktails": ToolDiscoveryMetadata("Cocktail recipe search", TOOL_DISCOVERY_CATEGORY_RECIPES, "Search cocktail recipes by name or keyword, including ingredients, classifications, glass types, and instructions."),
    "math_calculate": ToolDiscoveryMetadata("Mathematical calculation", TOOL_CATEGORY_MATH, "Evaluate arithmetic expressions and mathematical functions."),
    "code_execute": ToolDiscoveryMetadata("Sandboxed code execution", TOOL_DISCOVERY_CATEGORY_CODE_EXECUTION, "Execute Python code in an isolated Docker sandbox with no network access and resource limits. The conversation workspace is mounted at /workspace and persists for the current conversation.", version=2),
    "games_commander_details": ToolDiscoveryMetadata("Commander deck details", TOOL_DISCOVERY_CATEGORY_MAGIC, "Retrieve EDHREC commander themes, deck details, and combo information."),
    "games_commander_cards": ToolDiscoveryMetadata("Commander card recommendations", TOOL_DISCOVERY_CATEGORY_MAGIC, "Retrieve reranked card recommendations for a Magic: The Gathering commander."),
    "games_search_cards": ToolDiscoveryMetadata("Magic card search", TOOL_DISCOVERY_CATEGORY_MAGIC, "Search Scryfall for Magic: The Gathering cards by name or advanced query."),
    "games_card_rulings": ToolDiscoveryMetadata("Magic card rulings", TOOL_DISCOVERY_CATEGORY_MAGIC, "Retrieve official rules rulings for a Magic: The Gathering card."),
    "memory_detail": ToolDiscoveryMetadata("Conversation memory detail", TOOL_DISCOVERY_CATEGORY_MEMORIES, "Retrieve the exact prompt, response, or structured payload for a previously found conversation memory using the exact roundtrip UUID returned by memory_search. Do not pass a topic, title, conversation name, or search query.", version=4),
    "memory_lookup_evidence": ToolDiscoveryMetadata("Historical evidence lookup", TOOL_DISCOVERY_CATEGORY_MEMORIES, "Retrieve full evidence records by evidence ID from prior conversation context."),
    "memory_search": ToolDiscoveryMetadata("Conversation memory search", TOOL_DISCOVERY_CATEGORY_MEMORIES, "Discover relevant past conversations or search detailed exchanges within selected conversations."),
    "asset_search": ToolDiscoveryMetadata("Asset discovery", TOOL_DISCOVERY_CATEGORY_ASSETS, "Search assets by semantic relevance, asset type, or content type.", ("documents", "web pages", "asset metadata")),
    "asset_search_details": ToolDiscoveryMetadata("Asset detail search", TOOL_DISCOVERY_CATEGORY_ASSETS, "Search the contents of a known asset for specific details or passages.", ("asset content", "known asset")),
    "file_get": ToolDiscoveryMetadata("Retrieve uploaded file", TOOL_DISCOVERY_CATEGORY_UPLOADED_FILES, "Retrieve metadata and a content preview for a known uploaded file.", ("file ID", "document preview", "known file")),
    "file_create": ToolDiscoveryMetadata("Create a managed file", TOOL_DISCOVERY_CATEGORY_FILE_MANAGEMENT, "Create a new managed text, Word, PDF, or SVG file in txt, md, docx, pdf, or svg format.", ("new file", "write file", "Word document", "DOCX", "PDF", "SVG", "image"), version=6),
    "file_read": ToolDiscoveryMetadata("Read a managed file", TOOL_DISCOVERY_CATEGORY_FILE_MANAGEMENT, "Retrieve text content from a managed text, Word, or PDF file.", ("read file", "file contents", "DOCX", "PDF"), version=4),
    "file_edit": ToolDiscoveryMetadata("Edit a managed file", TOOL_DISCOVERY_CATEGORY_FILE_MANAGEMENT, "Apply a targeted text replacement to a managed text, Word, or PDF file.", ("modify file", "replace text", "DOCX", "PDF"), version=4),
    "file_delete": ToolDiscoveryMetadata("Delete a managed file", TOOL_DISCOVERY_CATEGORY_FILE_MANAGEMENT, "Remove a managed file and its indexed content.", ("remove file", "delete file")),
    "file_list": ToolDiscoveryMetadata("List managed files", TOOL_DISCOVERY_CATEGORY_FILE_MANAGEMENT, "Discover files in the managed files directory.", ("list files", "file inventory")),
    "file_move": ToolDiscoveryMetadata("Move a managed file", TOOL_DISCOVERY_CATEGORY_FILE_MANAGEMENT, "Rename or relocate a managed file within the managed files directory.", ("rename file", "move file")),
    "writing_templates": ToolDiscoveryMetadata("Writing templates", TOOL_CATEGORY_WRITING, "Discover reusable writing template metadata and supported output formats. Returns metadata only, not template content.", ("document template", "essay template", "report template", "meeting notes"), version=1),
    "workspace_file_read": ToolDiscoveryMetadata("Read workspace file", TOOL_CATEGORY_WORKSPACE, "Read text or source content from a file in the current conversation workspace without modifying it.", ("workspace", "file contents", "source code"), version=2),
    "workspace_file_list": ToolDiscoveryMetadata("Search workspace files", TOOL_CATEGORY_WORKSPACE, "List or search files in the current conversation workspace by name, regular expression, or text content without modifying them.", ("workspace", "find files", "search file contents"), version=3),
    "user_update_names": ToolDiscoveryMetadata("Update user names", TOOL_DISCOVERY_CATEGORY_USER_PROFILE, "Update any explicitly provided display, first, or last name fields; empty fields remain unchanged."),
    "user_update_tone": ToolDiscoveryMetadata("Update user tone preferences", TOOL_DISCOVERY_CATEGORY_USER_PROFILE, "Store durable preferences for response tone, directness, humor, and technical depth."),
}

def get_tool_discovery_metadata(tool_name: str) -> ToolDiscoveryMetadata:
    try:
        return DISCOVERY_METADATA[tool_name]
    except KeyError as exc:
        raise ValueError(f"Missing discovery metadata for tool '{tool_name}'.") from exc


def validate_tool_discovery_metadata(tool_names: set[str]) -> None:
    missing = sorted(tool_names - DISCOVERY_METADATA.keys())
    if missing:
        raise ValueError(f"Missing discovery metadata for tools: {', '.join(missing)}")
