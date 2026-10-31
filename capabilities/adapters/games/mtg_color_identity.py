from __future__ import annotations

import re

COMMANDER_COLOR_ALIASES = {
    "colorless": "c",
    "white": "w",
    "blue": "u",
    "black": "b",
    "red": "r",
    "green": "g",
    "azorius": "wu",
    "dimir": "ub",
    "rakdos": "br",
    "gruul": "rg",
    "selesnya": "gw",
    "orzhov": "wb",
    "izzet": "ur",
    "golgari": "bg",
    "boros": "rw",
    "simic": "ug",
    "esper": "wub",
    "grixis": "ubr",
    "jund": "brg",
    "naya": "rgw",
    "bant": "wug",
    "abzan": "wbg",
    "jeskai": "wur",
    "sultai": "ubg",
    "mardu": "rwb",
    "temur": "urg",
    "sanswhite": "ubr",
    "sansblue": "brg",
    "sansblack": "rgw",
    "sansred": "wug",
    "sansgreen": "wub",
    "fivecolor": "wubrg",
    "5color": "wubrg",
    "wubrg": "wubrg",
}
COMMANDER_IDENTITY_FILTER_PATTERN = re.compile(r"(^|\s)(id|identity|ci)\s*(<=|>=|=|:|<|>)", re.IGNORECASE)
COLOR_ORDER = "wubrgc"


def normalize_mtg_color_identity(value: str) -> str:
    normalized = "".join(ch for ch in (value or "").lower() if ch.isalpha() or ch.isdigit())
    if not normalized:
        return ""
    alias = COMMANDER_COLOR_ALIASES.get(normalized)
    if alias is not None:
        return alias

    deduped: list[str] = []
    for symbol in COLOR_ORDER:
        if symbol in normalized:
            deduped.append(symbol)
    if deduped:
        return "".join(deduped)
    return normalized


def query_has_color_identity_filter(query: str) -> bool:
    return bool(COMMANDER_IDENTITY_FILTER_PATTERN.search(query))


def apply_commander_color_identity_filter(query: str, commander_color_identity: str) -> str:
    normalized_query = query.strip()
    if not normalized_query:
        return normalized_query
    normalized_identity = normalize_mtg_color_identity(commander_color_identity)
    if not normalized_identity or query_has_color_identity_filter(normalized_query):
        return normalized_query
    return f"id<={normalized_identity} {normalized_query}"
