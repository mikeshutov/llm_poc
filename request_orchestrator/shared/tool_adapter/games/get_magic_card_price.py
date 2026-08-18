from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from integrations.scryfall import MagicCardPriceEntry, MagicCardPriceResult, ScryfallClient
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
    pricing = [MagicCardPriceEntry.from_card(card) for card in search_result.data]
    result = MagicCardPriceResult.from_card(resolved_card, pricing=pricing)
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
