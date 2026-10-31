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


_LEGACY_DISCOVERY_METADATA: dict[str, ToolDiscoveryMetadata] = {
    "products.search": ToolDiscoveryMetadata("Product search", "products", "Search internal catalog products, external shopping results, or both using an explicit source.", ("catalog", "inventory", "web shopping", "price")),
    "products.list_categories": ToolDiscoveryMetadata("Internal product category lookup", "internal products", "List categories available in the internal product catalog.", ("catalog", "categories")),
    "weather.current": ToolDiscoveryMetadata("Current weather lookup", "weather", "Retrieve current weather conditions for a city or location.", ("temperature", "wind", "conditions")),
    "weather.historical_month": ToolDiscoveryMetadata("Historical monthly weather lookup", "weather", "Retrieve historical weather conditions for a city, year, and month."),
    "finance.exchange_rates": ToolDiscoveryMetadata("Currency exchange rates", "finance", "Retrieve latest, historical-date, or date-range exchange rates between currencies."),
    "finance.stock_price": ToolDiscoveryMetadata("Stock market price lookup", "finance", "Retrieve current stock prices and basic market data by ticker."),
    "finance_crypto.markets": ToolDiscoveryMetadata("Cryptocurrency market lookup", "finance", "Retrieve cryptocurrency prices, market caps, volume, and market changes."),
    "search.web": ToolDiscoveryMetadata("General web search", "web search", "Search the public web for general information and current web content.", ("internet", "web pages")),
    "search.news": ToolDiscoveryMetadata("Current news search", "news", "Search current news stories and article overviews.", ("breaking news", "recent events")),
    "search.wikipedia": ToolDiscoveryMetadata("Wikipedia search", "knowledge", "Search Wikipedia pages and retrieve short article summaries."),
    "search.facts": ToolDiscoveryMetadata("Structured knowledge lookup", "knowledge", "Retrieve stable entity and relationship facts from Wikidata using structured queries.", ("SPARQL", "facts", "entities")),
    "search.hacker_news": ToolDiscoveryMetadata("Hacker News search", "news", "Search Hacker News stories, comments, and discussions."),
    "search.country": ToolDiscoveryMetadata("Country information lookup", "knowledge", "Retrieve country details such as capital, region, population, currencies, and languages."),
    "calendar.public_holidays": ToolDiscoveryMetadata("Public holiday lookup", "calendar", "Retrieve official public holidays for a country and year."),
    "calendar.world_time": ToolDiscoveryMetadata("World time lookup", "calendar", "Retrieve the current local date and time for an IANA timezone."),
    "location.caller": ToolDiscoveryMetadata("Caller location lookup", "location", "Retrieve the caller's approximate country, region, city, timezone, and coordinates from IP location."),
    "books.search": ToolDiscoveryMetadata("Book catalog search", "books", "Search the Open Library catalog by book title, author, subject, or keyword."),
    "language.define": ToolDiscoveryMetadata("Word definition lookup", "language", "Retrieve definitions, parts of speech, examples, synonyms, and antonyms for an English word."),
    "food.meals": ToolDiscoveryMetadata("Meal recipe search", "recipes", "Search meal recipes by name or keyword, including ingredients, categories, cuisine areas, and cooking instructions."),
    "food.cocktails": ToolDiscoveryMetadata("Cocktail recipe search", "recipes", "Search cocktail recipes by name or keyword, including ingredients, classifications, glass types, and instructions."),
    "fun.advice": ToolDiscoveryMetadata("Advice lookup", "fun content", "Retrieve random or topic-specific advice."),
    "fun.quote": ToolDiscoveryMetadata("Quote lookup", "fun content", "Retrieve random or topic-specific inspirational quotes."),
    "fun.astronomy_picture": ToolDiscoveryMetadata("Astronomy picture lookup", "astronomy", "Retrieve NASA's Astronomy Picture of the Day and its explanation."),
    "math.calculate": ToolDiscoveryMetadata("Mathematical calculation", "math", "Evaluate arithmetic expressions and mathematical functions."),
    "games.commander_details": ToolDiscoveryMetadata("Commander deck details", "Magic: The Gathering", "Retrieve EDHREC commander themes, deck details, and combo information."),
    "games.commander_cards": ToolDiscoveryMetadata("Commander card recommendations", "Magic: The Gathering", "Retrieve reranked card recommendations for a Magic: The Gathering commander."),
    "games.search_cards": ToolDiscoveryMetadata("Magic card search", "Magic: The Gathering", "Search Scryfall for Magic: The Gathering cards by name or advanced query."),
    "games.card_rulings": ToolDiscoveryMetadata("Magic card rulings", "Magic: The Gathering", "Retrieve official rules rulings for a Magic: The Gathering card."),
    "memory.detail": ToolDiscoveryMetadata("Conversation memory detail", "conversation memory", "Retrieve the exact prompt, response, or structured payload for a previously found conversation memory."),
    "memory.lookup_evidence": ToolDiscoveryMetadata("Historical evidence lookup", "conversation memory", "Retrieve full evidence records by evidence ID from prior conversation context."),
    "memory.search": ToolDiscoveryMetadata("Conversation memory search", "conversation memory", "Discover relevant past conversations or search detailed exchanges within selected conversations."),
    "user.create_attribute": ToolDiscoveryMetadata("Create user attribute", "persistent user profile", "Store a durable user fact, preference, interest, skill, goal, or other profile attribute."),
    "user.update_attribute": ToolDiscoveryMetadata("Update user attribute", "persistent user profile", "Update or deactivate an existing durable user profile attribute."),
    "user.get_attributes": ToolDiscoveryMetadata("List user attributes", "persistent user profile", "List stored durable user profile attributes."),
    "user.search_attributes": ToolDiscoveryMetadata("Search user attributes", "persistent user profile", "Search durable user profile attributes by semantic similarity."),
    "file.search": ToolDiscoveryMetadata("Uploaded file discovery", "uploaded files", "Discover uploaded files by filename, file type, or semantic relevance to a query.", ("documents", "resumes", "internal files", "file metadata")),
    "file.search_details": ToolDiscoveryMetadata("Search inside uploaded file", "uploaded files", "Search the contents of a known uploaded file for specific details or passages.", ("document content", "file text", "known file")),
    "file.get": ToolDiscoveryMetadata("Retrieve uploaded file", "uploaded files", "Retrieve metadata and a content preview for a known uploaded file.", ("file ID", "document preview", "known file")),
    "file.create": ToolDiscoveryMetadata("Create a managed file", "file management", "Create a new UTF-8 text file in the managed files directory.", ("new file", "write file")),
    "file.read": ToolDiscoveryMetadata("Read a managed file", "file management", "Retrieve the UTF-8 text content of a managed file.", ("read file", "file contents")),
    "file.edit": ToolDiscoveryMetadata("Edit a managed file", "file management", "Apply a targeted text replacement to a managed file.", ("modify file", "replace text")),
    "file.delete": ToolDiscoveryMetadata("Delete a managed file", "file management", "Remove a managed file and its indexed content.", ("remove file", "delete file")),
    "file.list": ToolDiscoveryMetadata("List managed files", "file management", "Discover files in the managed files directory.", ("list files", "file inventory")),
    "file.move": ToolDiscoveryMetadata("Move a managed file", "file management", "Rename or relocate a managed file within the managed files directory.", ("rename file", "move file")),
    "user.set_display_name": ToolDiscoveryMetadata("Set user display name", "persistent user profile", "Store the user's preferred nickname or display name."),
    "user.set_first_name": ToolDiscoveryMetadata("Set user first name", "persistent user profile", "Store the user's explicitly provided first name."),
    "user.set_last_name": ToolDiscoveryMetadata("Set user last name", "persistent user profile", "Store the user's explicitly provided last name."),
    "user.update_tone": ToolDiscoveryMetadata("Update user tone preferences", "persistent user profile", "Store durable preferences for response tone, directness, humor, and technical depth."),
}

# Keep the metadata definitions compact while exposing only provider-safe,
# underscore-based canonical names to the rest of the application.
DISCOVERY_METADATA: dict[str, ToolDiscoveryMetadata] = {
    name.replace(".", "_"): metadata
    for name, metadata in _LEGACY_DISCOVERY_METADATA.items()
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
