from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from integrations.edhrec import EdhrecClient, EdhrecCommanderPage, EdhrecComboLink
from request_orchestrator.models.evidence import EvidenceUrl, EvidenceView, HydratedEvidence, ToolResult
from tool.constants import TOOL_NAME_GET_COMMANDER_DETAILS
from tool.constants import TOOL_RESULT_TYPE_DECKS

_edhrec_client = EdhrecClient()


class GetCommanderDetailsArgs(BaseModel):
    commander_name: str = Field(
        ...,
        description="Commander name to look up on EDHREC. Example: 'Uril, the Miststalker'.",
    )
    theme_limit: int = Field(
        default=5,
        ge=1,
        le=12,
        description="Maximum number of top EDHREC themes to include.",
    )
    combo_limit: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum number of combo highlights to include.",
    )


class CommanderDetailsResult(BaseModel):
    query: str
    commander_name: str
    commander_slug: str
    page_url: str
    title: str
    description: str
    top_themes: str = ""
    combo_highlights: list[str] = []
    similar_commanders: list[str] = []


def _page_url(slug: str) -> str:
    return f"https://edhrec.com/commanders/{slug}"


def _combo_text(combo: EdhrecComboLink) -> str:
    return (combo.value or combo.alt or "").strip()


def _summary(result: CommanderDetailsResult) -> str:
    parts: list[str] = []
    if result.description.strip():
        parts.append(result.description.strip())
    if result.top_themes.strip():
        parts.append(f"Top themes: {result.top_themes}")
    if result.combo_highlights:
        parts.append("Combos: " + ", ".join(result.combo_highlights[:3]))
    if result.similar_commanders:
        parts.append("Similar commanders: " + ", ".join(result.similar_commanders[:4]))
    return " | ".join(parts) if parts else f"Commander details for {result.commander_name}."


def _tool_result(result: CommanderDetailsResult) -> ToolResult:
    metadata = {
        "query": result.query,
        "commander_slug": result.commander_slug,
        "top_themes": result.top_themes,
        "combo_highlights": list(result.combo_highlights),
        "similar_commanders": list(result.similar_commanders),
    }
    evidence_object = result.model_dump()
    hydrated = HydratedEvidence(
        item_id=result.commander_slug,
        tool_name=TOOL_NAME_GET_COMMANDER_DETAILS,
        title=result.title.strip() or result.commander_name,
        summary=_summary(result),
        urls=[EvidenceUrl(url=result.page_url, url_type="website")] if result.page_url else [],
        source=TOOL_NAME_GET_COMMANDER_DETAILS,
        entity_type=TOOL_RESULT_TYPE_DECKS,
        metadata=metadata,
        evidence_object=evidence_object,
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
                evidence_object=evidence_object,
            )
        ],
        hydrated_evidence=[hydrated],
    )


@tool(
    TOOL_NAME_GET_COMMANDER_DETAILS,
    args_schema=GetCommanderDetailsArgs,
    description="""
Look up EDHREC commander details for a Magic: The Gathering commander.

Required fields:
- commander_name (string)

Optional fields:
- theme_limit (integer, 1-12)
- combo_limit (integer, 0-10)
""",
)
def get_commander_details(commander_name: str, theme_limit: int = 5, combo_limit: int = 3) -> ToolResult:
    slug, page = _edhrec_client.get_commander_page(commander_name)
    top_themes = ", ".join(theme.value for theme in page.panels.taglinks[:theme_limit] if theme.value.strip())
    combo_highlights = [
        _combo_text(combo)
        for combo in page.panels.combocounts
        if _combo_text(combo) and not _combo_text(combo).lower().startswith("see more")
    ][:combo_limit]
    return _tool_result(
        CommanderDetailsResult(
            query=commander_name.strip(),
            commander_name=page.container.title.replace(" (Commander)", "").strip() or commander_name.strip(),
            commander_slug=slug,
            page_url=_page_url(slug),
            title=page.container.title.strip() or page.header.strip() or commander_name.strip(),
            description=(page.container.description or page.description or "").strip(),
            top_themes=top_themes,
            combo_highlights=combo_highlights,
            similar_commanders=list(page.similar),
        )
    )
