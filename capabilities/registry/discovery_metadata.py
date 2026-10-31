from __future__ import annotations

from dataclasses import dataclass


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


_DISCOVERY_METADATA: dict[str, ToolDiscoveryMetadata] = {
    "products_search": ToolDiscoveryMetadata("Product search", "products", "Search internal catalog products, external shopping results, or both using an explicit source.", ("catalog", "inventory", "web shopping", "price")),
    "products_list_categories": ToolDiscoveryMetadata("Internal product category lookup", "internal products", "List categories available in the internal product catalog.", ("catalog", "categories")),
    "weather_current": ToolDiscoveryMetadata("Current weather lookup", "weather", "Retrieve current weather conditions for a city or location.", ("temperature", "wind", "conditions")),
    "weather_historical_month": ToolDiscoveryMetadata("Historical monthly weather lookup", "weather", "Retrieve historical weather conditions for a city, year, and month."),
    "finance_exchange_rates": ToolDiscoveryMetadata("Currency exchange rates", "finance", "Retrieve latest, historical-date, or date-range exchange rates between currencies."),
    "finance_stock_price": ToolDiscoveryMetadata("Stock market price lookup", "finance", "Retrieve current stock prices and basic market data by ticker."),
    "finance_crypto_markets": ToolDiscoveryMetadata("Cryptocurrency market lookup", "finance", "Retrieve cryptocurrency prices, market caps, volume, and market changes."),
    "web_search": ToolDiscoveryMetadata("General web search", "web search", "Search the public web for general information and current web content.", ("internet", "web pages")),
    "web_fetch": ToolDiscoveryMetadata("Web page fetch", "web search", "Fetch a public web page and extract its readable title, description, and text for detailed research.", ("internet", "web pages", "article details")),
    "search_news": ToolDiscoveryMetadata("Current news search", "news", "Search current news stories and article overviews.", ("breaking news", "recent events")),
    "search_wikipedia": ToolDiscoveryMetadata("Wikipedia search", "knowledge", "Search Wikipedia pages and retrieve short article summaries."),
    "search_facts": ToolDiscoveryMetadata("Structured knowledge lookup", "knowledge", "Retrieve stable entity and relationship facts from Wikidata using structured queries.", ("SPARQL", "facts", "entities")),
    "search_hacker_news": ToolDiscoveryMetadata("Hacker News search", "news", "Search Hacker News stories, comments, and discussions."),
    "search_country": ToolDiscoveryMetadata("Country information lookup", "knowledge", "Retrieve country details such as capital, region, population, currencies, and languages."),
    "calendar_public_holidays": ToolDiscoveryMetadata("Public holiday lookup", "calendar", "Retrieve official public holidays for a country and year."),
    "calendar_world_time": ToolDiscoveryMetadata("World time lookup", "calendar", "Retrieve the current local date and time for an IANA timezone."),
    "location_caller": ToolDiscoveryMetadata("Caller location lookup", "location", "Retrieve the caller's approximate country, region, city, timezone, and coordinates from IP location."),
    "books_search": ToolDiscoveryMetadata("Book catalog search", "books", "Search the Open Library catalog by book title, author, subject, or keyword."),
    "language_define": ToolDiscoveryMetadata("Word definition lookup", "language", "Retrieve definitions, parts of speech, examples, synonyms, and antonyms for an English word."),
    "food_recipes": ToolDiscoveryMetadata("Recipe search", "recipes", "Search recipes by name or keyword, including ingredients, categories, cuisine areas, and cooking instructions."),
    "food_cocktails": ToolDiscoveryMetadata("Cocktail recipe search", "recipes", "Search cocktail recipes by name or keyword, including ingredients, classifications, glass types, and instructions."),
    "math_calculate": ToolDiscoveryMetadata("Mathematical calculation", "math", "Evaluate arithmetic expressions and mathematical functions."),
    "games_commander_details": ToolDiscoveryMetadata("Commander deck details", "Magic: The Gathering", "Retrieve EDHREC commander themes, deck details, and combo information."),
    "games_commander_cards": ToolDiscoveryMetadata("Commander card recommendations", "Magic: The Gathering", "Retrieve reranked card recommendations for a Magic: The Gathering commander."),
    "games_search_cards": ToolDiscoveryMetadata("Magic card search", "Magic: The Gathering", "Search Scryfall for Magic: The Gathering cards by name or advanced query."),
    "games_card_rulings": ToolDiscoveryMetadata("Magic card rulings", "Magic: The Gathering", "Retrieve official rules rulings for a Magic: The Gathering card."),
    "memory_detail": ToolDiscoveryMetadata("Conversation memory detail", "conversation memory", "Retrieve the exact prompt, response, or structured payload for a previously found conversation memory."),
    "memory_lookup_evidence": ToolDiscoveryMetadata("Historical evidence lookup", "conversation memory", "Retrieve full evidence records by evidence ID from prior conversation context."),
    "memory_search": ToolDiscoveryMetadata("Conversation memory search", "conversation memory", "Discover relevant past conversations or search detailed exchanges within selected conversations."),
    "asset_search": ToolDiscoveryMetadata("Asset discovery", "assets", "Search assets by semantic relevance, asset type, or content type.", ("documents", "web pages", "asset metadata")),
    "asset_search_details": ToolDiscoveryMetadata("Asset detail search", "assets", "Search the contents of a known asset for specific details or passages.", ("asset content", "known asset")),
    "file_get": ToolDiscoveryMetadata("Retrieve uploaded file", "uploaded files", "Retrieve metadata and a content preview for a known uploaded file.", ("file ID", "document preview", "known file")),
    "file_create": ToolDiscoveryMetadata("Create a managed file", "file management", "Create a new UTF-8 text file in the managed files directory.", ("new file", "write file")),
    "file_read": ToolDiscoveryMetadata("Read a managed file", "file management", "Retrieve the UTF-8 text content of a managed file.", ("read file", "file contents")),
    "file_edit": ToolDiscoveryMetadata("Edit a managed file", "file management", "Apply a targeted text replacement to a managed file.", ("modify file", "replace text")),
    "file_delete": ToolDiscoveryMetadata("Delete a managed file", "file management", "Remove a managed file and its indexed content.", ("remove file", "delete file")),
    "file_list": ToolDiscoveryMetadata("List managed files", "file management", "Discover files in the managed files directory.", ("list files", "file inventory")),
    "file_move": ToolDiscoveryMetadata("Move a managed file", "file management", "Rename or relocate a managed file within the managed files directory.", ("rename file", "move file")),
    "user_set_display_name": ToolDiscoveryMetadata("Set user display name", "persistent user profile", "Store the user's preferred nickname or display name."),
    "user_set_first_name": ToolDiscoveryMetadata("Set user first name", "persistent user profile", "Store the user's explicitly provided first name."),
    "user_set_last_name": ToolDiscoveryMetadata("Set user last name", "persistent user profile", "Store the user's explicitly provided last name."),
    "user_update_tone": ToolDiscoveryMetadata("Update user tone preferences", "persistent user profile", "Store durable preferences for response tone, directness, humor, and technical depth."),
}

DISCOVERY_METADATA: dict[str, ToolDiscoveryMetadata] = _DISCOVERY_METADATA


def get_tool_discovery_metadata(tool_name: str) -> ToolDiscoveryMetadata:
    try:
        return DISCOVERY_METADATA[tool_name]
    except KeyError as exc:
        raise ValueError(f"Missing discovery metadata for tool '{tool_name}'.") from exc


def validate_tool_discovery_metadata(tool_names: set[str]) -> None:
    missing = sorted(tool_names - DISCOVERY_METADATA.keys())
    if missing:
        raise ValueError(f"Missing discovery metadata for tools: {', '.join(missing)}")
