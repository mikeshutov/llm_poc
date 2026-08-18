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
    set_name: str | None = None
    rarity: str | None = None
    scryfall_uri: str | None = None
    image_url: str | None = None
    usd: str | None = None
    usd_foil: str | None = None
    usd_etched: str | None = None
    eur: str | None = None
    eur_foil: str | None = None
    tix: str | None = None


def _card_url(card: ScryfallCard) -> str:
    return (card.scryfall_uri or "").strip()


def _card_image_url(card: ScryfallCard) -> str:
    if card.image_uris is not None:
        return (card.image_uris.normal or card.image_uris.large or card.image_uris.small or "").strip()
    if not card.card_faces:
        return ""
    first_face = card.card_faces[0]
    if first_face.image_uris is None:
        return ""
    return (first_face.image_uris.normal or first_face.image_uris.large or first_face.image_uris.small or "").strip()


def _summary(result: MagicCardPriceResult) -> str:
    price_parts: list[str] = []
    if result.usd:
        price_parts.append(f"USD ${result.usd}")
    if result.usd_foil:
        price_parts.append(f"USD foil ${result.usd_foil}")
    if result.usd_etched:
        price_parts.append(f"USD etched ${result.usd_etched}")
    if result.eur:
        price_parts.append(f"EUR EUR {result.eur}")
    if result.eur_foil:
        price_parts.append(f"EUR foil EUR {result.eur_foil}")
    if result.tix:
        price_parts.append(f"TIX {result.tix}")
    if not price_parts:
        price_parts.append("No current price available.")
    if result.set_name:
        price_parts.append(f"Set: {result.set_name}")
    return " | ".join(price_parts)


def _tool_result(result: MagicCardPriceResult) -> ToolResult:
    metadata = {
        "set_name": result.set_name,
        "rarity": result.rarity,
        "usd": result.usd,
        "usd_foil": result.usd_foil,
        "usd_etched": result.usd_etched,
        "eur": result.eur,
        "eur_foil": result.eur_foil,
        "tix": result.tix,
    }
    hydrated = HydratedEvidence(
        item_id=result.id,
        tool_name=TOOL_NAME_GET_MAGIC_CARD_PRICE,
        title=f"{result.name} Price",
        summary=_summary(result),
        urls=[EvidenceUrl(url=result.scryfall_uri, url_type="website")] if result.scryfall_uri else [],
        image_url=result.image_url or "",
        source=TOOL_NAME_GET_MAGIC_CARD_PRICE,
        entity_type=TOOL_RESULT_TYPE_CARD_RESULTS,
        metadata=metadata,
        raw_payload=result,
    )
    return ToolResult(
        result=result,
        evidence_views=[
            EvidenceView(
                item_id=hydrated.item_id,
                title=hydrated.title,
                summary=hydrated.summary,
                metadata=dict(hydrated.metadata),
            )
        ],
        hydrated_evidence=[hydrated],
    )


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
    prices = dict(resolved_card.prices or {})
    return _tool_result(
        MagicCardPriceResult(
            id=resolved_card.id,
            name=resolved_card.name,
            set_name=resolved_card.set_name,
            rarity=resolved_card.rarity,
            scryfall_uri=resolved_card.scryfall_uri or None,
            image_url=_card_image_url(resolved_card) or None,
            usd=prices.get("usd"),
            usd_foil=prices.get("usd_foil"),
            usd_etched=prices.get("usd_etched"),
            eur=prices.get("eur"),
            eur_foil=prices.get("eur_foil"),
            tix=prices.get("tix"),
        )
    )
