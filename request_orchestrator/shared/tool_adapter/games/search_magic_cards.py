from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from integrations.scryfall import (
    MagicCardPriceEntry,
    MagicCardPriceMetadataEntry,
    ScryfallCard,
    ScryfallCardFace,
    ScryfallCardSearchResult,
    ScryfallClient,
)
from request_orchestrator.models.evidence import EvidenceUrl, EvidenceUrlType, EvidenceView, ToolMetadata, ToolResult
from request_orchestrator.shared.tool_adapter.games.mtg_color_identity import apply_commander_color_identity_filter
from tool.constants import TOOL_NAME_SEARCH_MAGIC_CARDS
from tool.constants import TOOL_RESULT_TYPE_CARD_RESULTS

_scryfall_client = ScryfallClient()
DEFAULT_MAGIC_CARD_PAGE_SIZE = 15


class SearchMagicCardsArgs(BaseModel):
    query: str = Field(
        ...,
        description="Scryfall card search query. Can be a card name or Scryfall search syntax like 'format:commander o:draw type:creature'.",
    )
    page: int = Field(default=1, ge=1, description="1-based page number for paginating card results.")
    commander_color_identity: str = Field(
        default="",
        description="Optional commander color identity constraint to apply with Scryfall `id<=...`, such as `naya`, `wug`, `abzan`, or `colorless`.",
    )
    include_pricing: bool = Field(
        default=False,
        description="Whether to include aggregated Scryfall pricing across printings for each returned card.",
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
    color_identity: list[str] | None = None
    image_url: str | None = None
    scryfall_uri: str | None = None
    set_name: str | None = None
    rarity: str | None = None
    pricing: list[MagicCardPriceEntry] = Field(default_factory=list)


class MagicCardSearchMetadata(BaseModel):
    rarity: str | None = None
    mana_cost: str | None = None
    type_line: str | None = None
    color_identity: list[str] = []
    legal_formats: list[str] = []
    pricing: list[MagicCardPriceMetadataEntry] | None = None


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


def _legal_formats(card: ScryfallCard) -> list[str]:
    return sorted(format_name for format_name, status in card.legalities.items() if status == "legal")


def _card_summary(card: ScryfallCard) -> str:
    oracle_text = _card_oracle_text(card)
    if oracle_text:
        return oracle_text.replace("\n", " ").strip()
    return f"Magic card result for {card.name}."


def _has_pricing(entry: MagicCardPriceEntry) -> bool:
    return any((entry.usd, entry.usd_foil, entry.usd_etched, entry.eur, entry.eur_foil, entry.tix))


def _load_card_pricing(card_name: str) -> list[MagicCardPriceEntry]:
    search_result = _scryfall_client.search_cards(
        f'!"{card_name}"',
        unique="prints",
        order="released",
        dir="desc",
    )
    return [MagicCardPriceEntry.from_card(card) for card in search_result.data]


def _build_card_record(card: ScryfallCard, pricing: list[MagicCardPriceEntry]) -> MagicCardSearchRecord:
    return MagicCardSearchRecord(
        id=card.id,
        name=card.name,
        mana_cost=card.mana_cost,
        cmc=card.cmc,
        type_line=card.type_line,
        oracle_text=card.oracle_text,
        color_identity=list(card.color_identity or []),
        image_url=_card_image_url(card) or None,
        scryfall_uri=card.scryfall_uri,
        set_name=card.set_name,
        rarity=card.rarity,
        pricing=pricing,
    )


def _build_evidence(card: ScryfallCard, pricing: list[MagicCardPriceEntry]) -> EvidenceView:
    pricing_metadata = [MagicCardPriceMetadataEntry.from_price_entry(entry) for entry in pricing if _has_pricing(entry)]
    metadata = MagicCardSearchMetadata(
        rarity=card.rarity,
        mana_cost=_card_mana_cost(card) or None,
        type_line=_card_type_line(card) or None,
        color_identity=list(card.color_identity or []),
        legal_formats=_legal_formats(card),
        pricing=pricing_metadata or None,
    )
    url = _card_url(card)
    return EvidenceView(
        item_id=card.id,
        tool_name=TOOL_NAME_SEARCH_MAGIC_CARDS,
        title=card.name.strip(),
        summary=_card_summary(card),
        urls=[EvidenceUrl(url=url, url_type=EvidenceUrlType.WEBSITE)] if url else [],
        image_url=_card_image_url(card) or "",
        source=TOOL_NAME_SEARCH_MAGIC_CARDS,
        entity_type=TOOL_RESULT_TYPE_CARD_RESULTS,
        llm_metadata=metadata.model_dump(exclude_none=True),
        raw_payload=card,
    )


def _build_search_query(query: str, commander_color_identity: str) -> str:
    return apply_commander_color_identity_filter(query, commander_color_identity)


@tool(
    TOOL_NAME_SEARCH_MAGIC_CARDS,
    args_schema=SearchMagicCardsArgs,
    description="""
Search Scryfall for Magic: The Gathering cards by name or advanced Scryfall query syntax.

Required fields:
- query (string)

Optional fields:
- page (integer, 1-based)
- commander_color_identity (string, optional)
- include_pricing (boolean, optional)

Example valid calls:
{
  "query": "Black Lotus",
  "page": 1
}
{
  "query": "format:commander o:draw type:creature",
  "page": 1
}
{
  "query": "type:instant (o:land or o:graveyard)",
  "commander_color_identity": "abzan",
  "page": 2
}
{
  "query": "Black Lotus",
  "include_pricing": true,
  "page": 1
}
""",
)
def search_magic_cards(
    query: str,
    page: int = 1,
    commander_color_identity: str = "",
    include_pricing: bool = False,
) -> ToolResult:
    response: ScryfallCardSearchResult = _scryfall_client.search_cards(
        _build_search_query(query, commander_color_identity)
    )
    start_index = (page - 1) * DEFAULT_MAGIC_CARD_PAGE_SIZE
    selected_cards = response.data[start_index : start_index + DEFAULT_MAGIC_CARD_PAGE_SIZE]
    pricing_by_name: dict[str, list[MagicCardPriceEntry]] = {}
    cards: list[MagicCardSearchRecord] = []
    evidence: list[EvidenceView] = []

    for card in selected_cards:
        pricing = pricing_by_name.setdefault(card.name, _load_card_pricing(card.name)) if include_pricing else []
        record = _build_card_record(card, pricing)
        evidence_view = _build_evidence(card, pricing)
        cards.append(record)
        evidence.append(evidence_view)

    result = SearchMagicCardsResult(
        total_cards=response.total_cards,
        has_more=response.total_cards > start_index + len(cards),
        returned_count=len(cards),
        warnings=list(response.warnings),
        cards=cards,
    )
    return ToolResult(
        result=result,
        tool_metadata=ToolMetadata(
            current_page=page,
            page_size=DEFAULT_MAGIC_CARD_PAGE_SIZE,
            has_more=result.has_more,
            returned_count=result.returned_count,
            warnings=list(result.warnings),
        ),

        evidence=evidence,
    )
