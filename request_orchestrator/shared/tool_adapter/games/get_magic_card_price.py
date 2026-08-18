from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from integrations.scryfall import ScryfallCard, ScryfallClient
from request_orchestrator.models.evidence import EvidenceUrl, EvidenceView, HydratedEvidence, ToolResult
from tool.constants import TOOL_NAME_GET_MAGIC_CARD_PRICE
from tool.constants import TOOL_RESULT_TYPE_CARD_RESULTS

_scryfall_client = ScryfallClient()


class GetMagicCardPriceArgs(BaseModel):
    card_name: str = Field(
        ...,
        description="Magic: The Gathering card name to look up pricing for.",
    )
    fuzzy: bool = Field(
        default=True,
        description="Whether to use Scryfall fuzzy name matching when resolving the card.",
    )


class MagicCardPriceResult(BaseModel):
    id: str
    name: str
    scryfall_uri: str | None = None
    image_url: str | None = None
    pricing: list["MagicCardPriceEntry"] = Field(default_factory=list)


class MagicCardPriceEntry(BaseModel):
    set_name: str | None = None
    scryfall_uri: str | None = None
    image_url: str | None = None
    usd: str | None = None
    usd_foil: str | None = None
    usd_etched: str | None = None
    eur: str | None = None
    eur_foil: str | None = None
    tix: str | None = None


def _card_image_url(card: ScryfallCard) -> str:
    if card.image_uris is not None:
        return (card.image_uris.normal or card.image_uris.large or card.image_uris.small or "").strip()
    if not card.card_faces:
        return ""
    first_face = card.card_faces[0]
    if first_face.image_uris is None:
        return ""
    return (first_face.image_uris.normal or first_face.image_uris.large or first_face.image_uris.small or "").strip()


@tool(
    TOOL_NAME_GET_MAGIC_CARD_PRICE,
    args_schema=GetMagicCardPriceArgs,
    description="""
Get current Scryfall pricing for a Magic: The Gathering card by name.

Required fields:
- card_name (string)

Optional fields:
- fuzzy (boolean)

Example valid calls:
{
  "card_name": "Black Lotus"
}
{
  "card_name": "Sol Ring",
  "fuzzy": false
}
""",
)
def get_magic_card_price(card_name: str, fuzzy: bool = True) -> ToolResult:
    resolved_card = _scryfall_client.get_card_by_name(card_name, fuzzy=fuzzy)
    search_result = _scryfall_client.search_cards(
        f'!"{resolved_card.name}"',
        unique="prints",
        order="released",
        dir="desc",
    )
    pricing = []
    for card in search_result.data:
        prices = dict(card.prices or {})
        pricing.append(
            MagicCardPriceEntry(
                set_name=card.set_name,
                scryfall_uri=card.scryfall_uri or None,
                image_url=_card_image_url(card) or None,
                usd=prices.get("usd"),
                usd_foil=prices.get("usd_foil"),
                usd_etched=prices.get("usd_etched"),
                eur=prices.get("eur"),
                eur_foil=prices.get("eur_foil"),
                tix=prices.get("tix"),
            )
        )

    result = MagicCardPriceResult(
        id=resolved_card.id,
        name=resolved_card.name,
        scryfall_uri=resolved_card.scryfall_uri or None,
        image_url=_card_image_url(resolved_card) or None,
        pricing=pricing,
    )
    priced_printings = sum(
        1
        for price in result.pricing
        if any((price.usd, price.usd_foil, price.usd_etched, price.eur, price.eur_foil, price.tix))
    )
    summary = "No current price available."
    if priced_printings:
        summary = f"Found pricing for {priced_printings} printings."

    hydrated = HydratedEvidence(
        item_id=result.id,
        tool_name=TOOL_NAME_GET_MAGIC_CARD_PRICE,
        title=f"{result.name} Price",
        summary=summary,
        urls=[EvidenceUrl(url=result.scryfall_uri, url_type="website")] if result.scryfall_uri else [],
        image_url=result.image_url or "",
        source=TOOL_NAME_GET_MAGIC_CARD_PRICE,
        entity_type=TOOL_RESULT_TYPE_CARD_RESULTS,
        metadata={},
        raw_payload=result,
    )
    return ToolResult(
        result=result,
        evidence_views=[
            EvidenceView(
                item_id=hydrated.item_id,
                title=hydrated.title,
                summary=hydrated.summary,
                metadata={},
            )
        ],
        hydrated_evidence=[hydrated],
    )
