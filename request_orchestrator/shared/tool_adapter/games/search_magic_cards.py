from __future__ import annotations

import re

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from integrations.scryfall import ScryfallCard, ScryfallCardFace, ScryfallCardSearchResult, ScryfallClient
from request_orchestrator.models.evidence import EvidenceUrl, EvidenceView, HydratedEvidence, ToolResult
from tool.constants import TOOL_NAME_SEARCH_MAGIC_CARDS
from tool.constants import TOOL_RESULT_TYPE_CARD_RESULTS

_scryfall_client = ScryfallClient()
_COMMANDER_COLOR_ALIASES = {
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
_COMMANDER_IDENTITY_FILTER_PATTERN = re.compile(r"(^|\s)(id|identity|ci)\s*(<=|>=|=|:|<|>)", re.IGNORECASE)
_COLOR_ORDER = "wubrgc"


class SearchMagicCardsArgs(BaseModel):
    query: str = Field(
        ...,
        description="Scryfall card search query. Can be a card name or Scryfall search syntax like 'format:commander o:draw type:creature'.",
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=25,
        description="Maximum number of card results to return.",
    )
    commander_color_identity: str = Field(
        default="",
        description="Optional commander color identity constraint to apply with Scryfall `id<=...`, such as `naya`, `wug`, `abzan`, or `colorless`.",
    )


class SearchMagicCardsResult(BaseModel):
    total_cards: int
    has_more: bool
    returned_count: int
    warnings: list[str] = []
    cards: list["MagicCardSearchRecord"] = []


class MagicCardSearchRecord(BaseModel):
    id: str
    name: str
    mana_cost: str | None = None
    cmc: float | None = None
    type_line: str | None = None
    oracle_text: str | None = None
    colors: list[str] | None = None
    color_identity: list[str] | None = None
    image_url: str | None = None
    scryfall_uri: str | None = None
    set_name: str | None = None
    rarity: str | None = None


def _first_face(card: ScryfallCard) -> ScryfallCardFace | None:
    if card.card_faces:
        return card.card_faces[0]
    return None


def _card_url(card: ScryfallCard) -> str:
    return (card.scryfall_uri or "").strip()


def _card_image_url(card: ScryfallCard) -> str:
    if card.image_uris is not None:
        return (card.image_uris.normal or card.image_uris.large or card.image_uris.small or "").strip()
    face = _first_face(card)
    if face is None or face.image_uris is None:
        return ""
    return (face.image_uris.normal or face.image_uris.large or face.image_uris.small or "").strip()


def _card_oracle_text(card: ScryfallCard) -> str:
    if card.oracle_text:
        return card.oracle_text.strip()
    if not card.card_faces:
        return ""
    parts = [part.strip() for part in ((face.oracle_text or "") for face in card.card_faces) if part.strip()]
    return " // ".join(parts)


def _card_type_line(card: ScryfallCard) -> str:
    if card.type_line:
        return card.type_line.strip()
    face = _first_face(card)
    return "" if face is None or not face.type_line else face.type_line.strip()


def _card_mana_cost(card: ScryfallCard) -> str:
    if card.mana_cost:
        return card.mana_cost.strip()
    face = _first_face(card)
    return "" if face is None or not face.mana_cost else face.mana_cost.strip()


def _card_summary(card: ScryfallCard) -> str:
    parts: list[str] = []
    mana_cost = _card_mana_cost(card)
    type_line = _card_type_line(card)
    oracle_text = _card_oracle_text(card)
    if mana_cost:
        parts.append(mana_cost)
    if type_line:
        parts.append(type_line)
    if oracle_text:
        parts.append(oracle_text.replace("\n", " ").strip())
    if card.set_name:
        parts.append(f"Set: {card.set_name}")
    return " | ".join(parts) if parts else f"Magic card result for {card.name}."


def _search_record_summary(card: MagicCardSearchRecord) -> str:
    parts: list[str] = []
    if card.mana_cost:
        parts.append(card.mana_cost.strip())
    if card.type_line:
        parts.append(card.type_line.strip())
    if card.oracle_text:
        parts.append(card.oracle_text.replace("\n", " ").strip())
    if card.set_name:
        parts.append(f"Set: {card.set_name}")
    return " | ".join(parts) if parts else f"Magic card result for {card.name}."


def _tool_result(result: SearchMagicCardsResult) -> ToolResult:
    hydrated_evidence: list[HydratedEvidence] = []
    evidence_views: list[EvidenceView] = []
    for card in result.cards:
        metadata = {
            "set_name": card.set_name,
            "rarity": card.rarity,
            "mana_cost": card.mana_cost or "",
            "type_line": card.type_line or "",
            "colors": list(card.colors or []),
            "color_identity": list(card.color_identity or []),
        }
        hydrated = HydratedEvidence(
            item_id=card.id,
            tool_name=TOOL_NAME_SEARCH_MAGIC_CARDS,
            title=card.name.strip(),
            summary=_search_record_summary(card),
            urls=[EvidenceUrl(url=(card.scryfall_uri or "").strip(), url_type="website")] if (card.scryfall_uri or "").strip() else [],
            image_url=(card.image_url or "").strip(),
            source=TOOL_NAME_SEARCH_MAGIC_CARDS,
            entity_type=TOOL_RESULT_TYPE_CARD_RESULTS,
            metadata=metadata,
            raw_payload=card,
        )
        hydrated_evidence.append(hydrated)
        evidence_views.append(
            EvidenceView(
                item_id=hydrated.item_id,
                title=hydrated.title,
                summary=hydrated.summary,
                metadata=dict(hydrated.metadata),
            )
        )
    return ToolResult(
        result=result,
        metadata={
            "has_more": result.has_more,
            "returned_count": result.returned_count,
            "total_cards": result.total_cards,
            "warnings": list(result.warnings),
        },
        evidence_views=evidence_views,
        hydrated_evidence=hydrated_evidence,
    )


def _normalize_commander_color_identity(value: str) -> str:
    normalized = "".join(ch for ch in (value or "").lower() if ch.isalpha() or ch.isdigit())
    if not normalized:
        return ""
    alias = _COMMANDER_COLOR_ALIASES.get(normalized)
    if alias is not None:
        return alias

    deduped: list[str] = []
    for symbol in _COLOR_ORDER:
        if symbol in normalized:
            deduped.append(symbol)
    if deduped:
        return "".join(deduped)
    return normalized


def _build_search_query(query: str, commander_color_identity: str) -> str:
    normalized_query = query.strip()
    if not normalized_query:
        return normalized_query
    normalized_identity = _normalize_commander_color_identity(commander_color_identity)
    if not normalized_identity or _COMMANDER_IDENTITY_FILTER_PATTERN.search(normalized_query):
        return normalized_query
    return f"id<={normalized_identity} {normalized_query}"


@tool(
    TOOL_NAME_SEARCH_MAGIC_CARDS,
    args_schema=SearchMagicCardsArgs,
    description="""
Search Scryfall for Magic: The Gathering cards by name or advanced Scryfall query syntax.

Required fields:
- query (string)

Optional fields:
- limit (integer, 1-25)
- commander_color_identity (string, optional)

Example valid calls:
{
  "query": "Black Lotus",
  "limit": 3
}
{
  "query": "format:commander o:draw type:creature",
  "limit": 5
}
{
  "query": "type:instant (o:land or o:graveyard)",
  "commander_color_identity": "abzan",
  "limit": 5
}
""",
)
def search_magic_cards(query: str, limit: int = 5, commander_color_identity: str = "") -> ToolResult:
    response: ScryfallCardSearchResult = _scryfall_client.search_cards(
        _build_search_query(query, commander_color_identity)
    )
    cards = [
        MagicCardSearchRecord(
            id=card.id,
            name=card.name,
            mana_cost=card.mana_cost,
            cmc=card.cmc,
            type_line=card.type_line,
            oracle_text=card.oracle_text,
            colors=list(card.colors or []),
            color_identity=list(card.color_identity or []),
            image_url=_card_image_url(card) or None,
            scryfall_uri=card.scryfall_uri,
            set_name=card.set_name,
            rarity=card.rarity,
        )
        for card in response.data[:limit]
    ]
    return _tool_result(
        SearchMagicCardsResult(
            total_cards=response.total_cards,
            has_more=response.has_more or response.total_cards > len(cards),
            returned_count=len(cards),
            warnings=list(response.warnings),
            cards=cards,
        )
    )
