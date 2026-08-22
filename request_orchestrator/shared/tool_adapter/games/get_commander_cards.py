from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from integrations.edhrec import EDHREC_CARD_URL_TEMPLATE, EdhrecCardView, EdhrecClient, EdhrecCommanderPage
from request_orchestrator.models.evidence import EvidenceUrl, EvidenceView, HydratedEvidence, ToolResult
from request_orchestrator.shared.tool_adapter.games.candidate_mapper import rerank_edhrec_cards
from tool.constants import TOOL_NAME_GET_COMMANDER_CARDS
from tool.constants import TOOL_RESULT_TYPE_CARD_RESULTS

_edhrec_client = EdhrecClient()


class GetCommanderCardsArgs(BaseModel):
    commander_name: str = Field(
        ...,
        description="Commander name to look up on EDHREC. Example: 'Uril, the Miststalker'.",
    )
    limit: int = Field(
        default=6,
        ge=1,
        le=12,
        description="Maximum number of EDHREC card recommendations to return after reranking.",
    )


class CommanderCardResult(BaseModel):
    name: str
    slug: str = ""
    card_url: str = ""
    section: str = ""
    synergy: float | None = None
    num_decks: int = 0
    potential_decks: int = 0
    trend_zscore: float | None = None


class CommanderCardsResult(BaseModel):
    query: str
    commander_name: str
    commander_slug: str
    returned_count: int
    cards: list[CommanderCardResult] = []


class CommanderCardMetadata(BaseModel):
    commander_slug: str
    section: str | None = None
    synergy: float | None = None
    num_decks: int | None = None
    potential_decks: int | None = None
    trend_zscore: float | None = None
    returned_count: int


def _flatten_candidate_cards(page: EdhrecCommanderPage) -> list[tuple[str, EdhrecCardView]]:
    flattened: list[tuple[str, EdhrecCardView]] = []
    for cardlist in page.container.json_dict.cardlists:
        section = cardlist.header.strip() or cardlist.tag.strip() or "EDHREC Cards"
        for card in cardlist.cardviews:
            if not card.name.strip():
                continue
            flattened.append((section, card))
    return flattened


def _card_result(section: str, card: EdhrecCardView) -> CommanderCardResult:
    return CommanderCardResult(
        name=card.name.strip(),
        slug=(card.slug or "").strip(),
        card_url=EDHREC_CARD_URL_TEMPLATE.format(path=card.url).strip() if card.url else "",
        section=section,
        synergy=card.synergy,
        num_decks=card.num_decks,
        potential_decks=card.potential_decks,
        trend_zscore=card.trend_zscore,
    )


def _card_summary(card: CommanderCardResult) -> str:
    parts = [card.section] if card.section else []
    if card.synergy is not None:
        parts.append(f"Synergy {card.synergy:.3f}")
    if card.num_decks:
        parts.append(f"{card.num_decks} decks")
    if card.potential_decks:
        parts.append(f"{card.potential_decks} potential decks")
    return " | ".join(parts) if parts else f"Commander card recommendation for {card.name}."


def _tool_result(result: CommanderCardsResult) -> ToolResult:
    hydrated_evidence: list[HydratedEvidence] = []
    evidence_views: list[EvidenceView] = []
    for card in result.cards:
        metadata = CommanderCardMetadata(
            commander_slug=result.commander_slug,
            section=card.section or None,
            synergy=card.synergy,
            num_decks=card.num_decks or None,
            potential_decks=card.potential_decks or None,
            trend_zscore=card.trend_zscore,
            returned_count=result.returned_count,
        )
        evidence_object = card.model_dump()
        hydrated = HydratedEvidence(
            item_id=card.slug or card.name,
            tool_name=TOOL_NAME_GET_COMMANDER_CARDS,
            title=card.name,
            summary=_card_summary(card),
            urls=[EvidenceUrl(url=card.card_url, url_type="website")] if card.card_url else [],
            source=TOOL_NAME_GET_COMMANDER_CARDS,
            entity_type=TOOL_RESULT_TYPE_CARD_RESULTS,
            metadata=metadata.model_dump(exclude_none=True),
            evidence_object=evidence_object,
            raw_payload=card,
        )
        hydrated_evidence.append(hydrated)
        evidence_views.append(
            EvidenceView(
                item_id=hydrated.item_id,
                title=hydrated.title,
                summary=hydrated.summary,
                metadata=dict(hydrated.metadata),
                evidence_object=evidence_object,
            )
        )
    return ToolResult(result=result, evidence_views=evidence_views, hydrated_evidence=hydrated_evidence)


@tool(
    TOOL_NAME_GET_COMMANDER_CARDS,
    args_schema=GetCommanderCardsArgs,
    description="""
Get reranked EDHREC card recommendations for a Magic: The Gathering commander.

Required fields:
- commander_name (string)

Optional fields:
- limit (integer, 1-12)
""",
)
def get_commander_cards(commander_name: str, limit: int = 6) -> ToolResult:
    slug, page = _edhrec_client.get_commander_page(commander_name)
    ranked_cards = rerank_edhrec_cards(
        _flatten_candidate_cards(page),
        goal=f"Most relevant EDHREC card recommendations for a {commander_name.strip()} commander deck.",
        limit=limit,
    )
    return _tool_result(
        CommanderCardsResult(
            query=commander_name.strip(),
            commander_name=page.container.title.replace(" (Commander)", "").strip() or commander_name.strip(),
            commander_slug=slug,
            returned_count=len(ranked_cards),
            cards=[_card_result(section, card) for section, card in ranked_cards],
        )
    )
