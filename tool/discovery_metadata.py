from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolDiscoveryMetadata:
    title: str
    category: str
    description: str
    scope_terms: tuple[str, ...] = ()
    version: int = 2

    def embedding_text(self, tool_name: str) -> str:
        parts = [tool_name, self.title, self.category, self.description]
        if self.scope_terms:
            parts.append("Scope: " + ", ".join(self.scope_terms))
        return "\n".join(parts)


DISCOVERY_METADATA: dict[str, ToolDiscoveryMetadata] = {
    "find_products": ToolDiscoveryMetadata("Internal product catalog search", "internal products", "Search products, inventory, prices, categories, and product attributes in the internal catalog.", ("catalog", "inventory", "shopping filters", "price")),
    "list_product_categories": ToolDiscoveryMetadata("Internal product category lookup", "internal products", "List categories available in the internal product catalog.", ("catalog", "categories")),
    "find_products_web": ToolDiscoveryMetadata("External web product search", "external products", "Search external websites for products and shopping listings when the internal catalog is insufficient.", ("web shopping", "retail listings", "external catalog")),
    "resolve_city_location": ToolDiscoveryMetadata("City location resolution", "location and weather", "Resolve a city name into normalized geographic location data for weather and location-aware decisions."),
    "get_current_weather": ToolDiscoveryMetadata("Current weather lookup", "weather", "Retrieve current weather conditions for a city or location.", ("temperature", "wind", "conditions")),
    "get_historical_month_weather": ToolDiscoveryMetadata("Historical monthly weather lookup", "weather", "Retrieve historical weather conditions for a city, year, and month."),
    "exchange_rates_lookup": ToolDiscoveryMetadata("Currency exchange rate lookup", "finance", "Retrieve current or historical exchange rates between currencies."),
    "exchange_rates_time_series": ToolDiscoveryMetadata("Historical exchange-rate time series", "finance", "Retrieve exchange-rate changes across a date range."),
    "get_latest_exchange_rates": ToolDiscoveryMetadata("Latest exchange rates", "finance", "Retrieve the latest rates for a base currency against supported currencies."),
    "get_stock_price": ToolDiscoveryMetadata("Stock market price lookup", "finance", "Retrieve current stock prices and basic market data by ticker."),
    "get_crypto_markets": ToolDiscoveryMetadata("Cryptocurrency market lookup", "finance", "Retrieve cryptocurrency prices, market caps, volume, and market changes."),
    "generic_web_search": ToolDiscoveryMetadata("General web search", "web search", "Search the public web for general information and current web content.", ("internet", "web pages")),
    "news_search": ToolDiscoveryMetadata("Current news search", "news", "Search current news stories and article overviews.", ("breaking news", "recent events")),
    "wikipedia_search": ToolDiscoveryMetadata("Wikipedia search", "knowledge", "Search Wikipedia pages and retrieve short article summaries."),
    "structured_facts_lookup": ToolDiscoveryMetadata("Structured knowledge lookup", "knowledge", "Retrieve stable entity and relationship facts from Wikidata using structured queries.", ("SPARQL", "facts", "entities")),
    "hn_search": ToolDiscoveryMetadata("Hacker News search", "news", "Search Hacker News stories, comments, and discussions."),
    "country_lookup": ToolDiscoveryMetadata("Country information lookup", "knowledge", "Retrieve country details such as capital, region, population, currencies, and languages."),
    "public_holidays_lookup": ToolDiscoveryMetadata("Public holiday lookup", "calendar", "Retrieve official public holidays for a country and year."),
    "get_world_time": ToolDiscoveryMetadata("World time lookup", "calendar", "Retrieve the current local date and time for an IANA timezone."),
    "get_caller_location": ToolDiscoveryMetadata("Caller location lookup", "location", "Retrieve the caller's approximate country, region, city, timezone, and coordinates from IP location."),
    "search_books": ToolDiscoveryMetadata("Book catalog search", "books", "Search the Open Library catalog by book title, author, subject, or keyword."),
    "define_word": ToolDiscoveryMetadata("Word definition lookup", "language", "Retrieve definitions, parts of speech, examples, synonyms, and antonyms for an English word."),
    "search_meals": ToolDiscoveryMetadata("Meal recipe search", "recipes", "Search meal recipes by name or keyword, including ingredients, categories, cuisine areas, and cooking instructions."),
    "search_cocktails": ToolDiscoveryMetadata("Cocktail recipe search", "recipes", "Search cocktail recipes by name or keyword, including ingredients, classifications, glass types, and instructions."),
    "get_advice": ToolDiscoveryMetadata("Advice lookup", "fun content", "Retrieve random or topic-specific advice."),
    "get_quote": ToolDiscoveryMetadata("Quote lookup", "fun content", "Retrieve random or topic-specific inspirational quotes."),
    "get_astronomy_picture": ToolDiscoveryMetadata("Astronomy picture lookup", "astronomy", "Retrieve NASA's Astronomy Picture of the Day and its explanation."),
    "calculate": ToolDiscoveryMetadata("Mathematical calculation", "math", "Evaluate arithmetic expressions and mathematical functions."),
    "get_commander_details": ToolDiscoveryMetadata("Commander deck details", "Magic: The Gathering", "Retrieve EDHREC commander themes, deck details, and combo information."),
    "get_commander_cards": ToolDiscoveryMetadata("Commander card recommendations", "Magic: The Gathering", "Retrieve reranked card recommendations for a Magic: The Gathering commander."),
    "search_magic_cards": ToolDiscoveryMetadata("Magic card search", "Magic: The Gathering", "Search Scryfall for Magic: The Gathering cards by name or advanced query."),
    "get_magic_card_rulings": ToolDiscoveryMetadata("Magic card rulings", "Magic: The Gathering", "Retrieve official rules rulings for a Magic: The Gathering card."),
    "get_memory_detail": ToolDiscoveryMetadata("Conversation memory detail", "conversation memory", "Retrieve the exact prompt, response, or structured payload for a previously found conversation memory."),
    "lookup_evidence": ToolDiscoveryMetadata("Historical evidence lookup", "conversation memory", "Retrieve full evidence records by evidence ID from prior conversation context."),
    "search_memories": ToolDiscoveryMetadata("Conversation memory search", "conversation memory", "Search prior conversation summaries for relevant past requests and discussions."),
    "search_roundtrip_memories": ToolDiscoveryMetadata("Conversation exchange search", "conversation memory", "Search specific historical exchanges within previously identified conversations."),
    "create_user_attribute": ToolDiscoveryMetadata("Create user attribute", "persistent user profile", "Store a durable user fact, preference, interest, skill, goal, or other profile attribute."),
    "update_user_attribute": ToolDiscoveryMetadata("Update user attribute", "persistent user profile", "Update or deactivate an existing durable user profile attribute."),
    "get_user_attributes": ToolDiscoveryMetadata("List user attributes", "persistent user profile", "List stored durable user profile attributes."),
    "search_user_attributes": ToolDiscoveryMetadata("Search user attributes", "persistent user profile", "Search durable user profile attributes by semantic similarity."),
    "search_files": ToolDiscoveryMetadata("Uploaded file discovery", "uploaded files", "Discover uploaded files by filename, file type, or semantic relevance to a query.", ("documents", "resumes", "internal files", "file metadata")),
    "search_file_for_details": ToolDiscoveryMetadata("Search inside uploaded file", "uploaded files", "Search the contents of a known uploaded file for specific details or passages.", ("document content", "file text", "known file")),
    "get_file_by_id": ToolDiscoveryMetadata("Retrieve uploaded file", "uploaded files", "Retrieve metadata and a content preview for a known uploaded file.", ("file ID", "document preview", "known file")),
    "set_user_display_name": ToolDiscoveryMetadata("Set user display name", "persistent user profile", "Store the user's preferred nickname or display name."),
    "set_user_first_name": ToolDiscoveryMetadata("Set user first name", "persistent user profile", "Store the user's explicitly provided first name."),
    "set_user_last_name": ToolDiscoveryMetadata("Set user last name", "persistent user profile", "Store the user's explicitly provided last name."),
    "update_user_tone": ToolDiscoveryMetadata("Update user tone preferences", "persistent user profile", "Store durable preferences for response tone, directness, humor, and technical depth."),
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
