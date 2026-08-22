from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from integrations.scryfall import ScryfallCard, ScryfallClient, ScryfallRuling
from request_orchestrator.models.evidence import EvidenceUrl, EvidenceUrlType, EvidenceView, HydratedEvidence, ToolResult
from tool.constants import TOOL_NAME_GET_MAGIC_CARD_RULINGS
from tool.constants import TOOL_RESULT_TYPE_RULES

_scryfall_client = ScryfallClient()


class GetMagicCardRulingsArgs(BaseModel):
    card_name: str = Field(
        ...,
        description="Magic: The Gathering card name to look up rulings for.",
    )
    fuzzy: bool = Field(
        default=True,
        description="Whether to use Scryfall fuzzy name matching when resolving the card.",
    )


class MagicCardRulingsResult(BaseModel):
    resolved_card: "MagicCardReference"
    ruling_count: int
    rulings: list[ScryfallRuling] = []


class MagicCardReference(BaseModel):
    id: str
    name: str
    type_line: str | None = None
    oracle_text: str | None = None
    scryfall_uri: str | None = None
    set_name: str | None = None
    rarity: str | None = None
    legal_formats: list[str] = []


class MagicCardRulingMetadata(BaseModel):
    legal_formats: list[str] = []
    ruling_count: int


class MagicCardRulingsMetadata(BaseModel):
    ruling_source: str | None = None


def _card_url(card: ScryfallCard) -> str:
    return (card.scryfall_uri or "").strip()


def _build_metadata(
    result: MagicCardRulingsResult,
    ruling: ScryfallRuling | None = None,
) -> MagicCardRulingMetadata:
    return MagicCardRulingMetadata(
        legal_formats=list(result.resolved_card.legal_formats),
        ruling_count=result.ruling_count,
    )


def _tool_metadata(result: MagicCardRulingsResult) -> dict[str, object]:
    sources = {ruling.source.strip() for ruling in result.rulings if ruling.source.strip()}
    source = next(iter(sources)) if len(sources) == 1 else None
    return MagicCardRulingsMetadata(ruling_source=source).model_dump(exclude_none=True)


def _tool_result(result: MagicCardRulingsResult) -> ToolResult:
    hydrated_evidence: list[HydratedEvidence] = []
    evidence_views: list[EvidenceView] = []
    card_url = _card_url(result.resolved_card)

    if not result.rulings:
        hydrated = HydratedEvidence(
            item_id=result.resolved_card.id,
            tool_name=TOOL_NAME_GET_MAGIC_CARD_RULINGS,
            title=f"{result.resolved_card.name} Rulings",
            summary=f"No published rulings found for {result.resolved_card.name}.",
            urls=[EvidenceUrl(url=card_url, url_type=EvidenceUrlType.WEBSITE)] if card_url else [],
            source=TOOL_NAME_GET_MAGIC_CARD_RULINGS,
            entity_type=TOOL_RESULT_TYPE_RULES,
            metadata=_build_metadata(result).model_dump(exclude_none=True),
            raw_payload=result.resolved_card,
        )
        return ToolResult(
            result=result,
            metadata=_tool_metadata(result),
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

    for index, ruling in enumerate(result.rulings, start=1):
        hydrated = HydratedEvidence(
            item_id=f"{result.resolved_card.id}:ruling:{index}",
            tool_name=TOOL_NAME_GET_MAGIC_CARD_RULINGS,
            title=f"{result.resolved_card.name} Ruling {index}",
            summary=ruling.comment.strip() or f"Ruling {index} for {result.resolved_card.name}.",
            urls=[EvidenceUrl(url=card_url, url_type=EvidenceUrlType.WEBSITE)] if card_url else [],
            published_at=ruling.published_at,
            source=TOOL_NAME_GET_MAGIC_CARD_RULINGS,
            entity_type=TOOL_RESULT_TYPE_RULES,
            metadata=_build_metadata(result, ruling).model_dump(exclude_none=True),
            raw_payload=ruling,
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
        metadata=_tool_metadata(result),
        evidence_views=evidence_views,
        hydrated_evidence=hydrated_evidence,
    )


@tool(
    TOOL_NAME_GET_MAGIC_CARD_RULINGS,
    args_schema=GetMagicCardRulingsArgs,
    description="""
Get official Scryfall card rulings for a Magic: The Gathering card by name.

Required fields:
- card_name (string)

Optional fields:
- fuzzy (boolean)

Example valid calls:
{
  "card_name": "Humility"
}
{
  "card_name": "Black Lotus",
  "fuzzy": false
}
""",
)
def get_magic_card_rulings(card_name: str, fuzzy: bool = True) -> ToolResult:
    resolved_card = _scryfall_client.get_card_by_name(card_name, fuzzy=fuzzy)
    rulings_response = _scryfall_client.get_card_rulings(resolved_card.id)
    return _tool_result(
        MagicCardRulingsResult(
            resolved_card=MagicCardReference(
                id=resolved_card.id,
                name=resolved_card.name,
                type_line=resolved_card.type_line,
                oracle_text=resolved_card.oracle_text,
                scryfall_uri=resolved_card.scryfall_uri,
                set_name=resolved_card.set_name,
                rarity=resolved_card.rarity,
                legal_formats=sorted(
                    format_name
                    for format_name, status in resolved_card.legalities.items()
                    if status == "legal"
                ),
            ),
            ruling_count=len(rulings_response.data),
            rulings=list(rulings_response.data),
        )
    )
