from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from requests.exceptions import RequestException

from integrations.cocktail_db import CocktailDbClient
from integrations.cocktail_db.models import CocktailSearchResult
from request_orchestrator.models.evidence import EvidenceView, HydratedEvidence, ToolResult
from request_orchestrator.shared.tool_adapter.food.candidate_mapper import rerank_cocktail_search_result
from request_orchestrator.shared.tool_adapter.food.constants import DEFAULT_MEAL_RERANK_LIMIT
from tool.constants import TOOL_NAME_SEARCH_COCKTAILS
from tool.constants import TOOL_RESULT_TYPE_COCKTAIL_RESULTS

_cocktail_client = CocktailDbClient()


class SearchCocktailsArgs(BaseModel):
    query: str = Field(
        ...,
        description="Cocktail name or keyword to search for. Example: 'margarita', 'mojito', 'gin'.",
    )


class CocktailSearchMetadata(BaseModel):
    category: str | None = None
    alcoholic: str | None = None
    glass: str | None = None
    tags: list[str] = []
    ingredients: list[dict[str, object]] = []


def _tool_result(result: CocktailSearchResult) -> ToolResult:
    hydrated_evidence: list[HydratedEvidence] = []
    evidence_views: list[EvidenceView] = []
    for cocktail in result.drinks:
        summary_parts = [
            part
            for part in (
                (cocktail.category or "").strip(),
                (cocktail.alcoholic or "").strip(),
                (cocktail.glass or "").strip(),
            )
            if part
        ]
        summary = ". ".join(summary_parts) or (cocktail.instructions or "").strip() or f"Cocktail result for {cocktail.name}."
        metadata = CocktailSearchMetadata(
            category=cocktail.category,
            alcoholic=cocktail.alcoholic,
            glass=cocktail.glass,
            tags=list(cocktail.tags or []),
            ingredients=[ingredient.model_dump(exclude_none=True) for ingredient in cocktail.ingredients],
        )
        hydrated = HydratedEvidence(
            item_id=cocktail.id,
            tool_name=TOOL_NAME_SEARCH_COCKTAILS,
            title=cocktail.name,
            summary=summary,
            image_url=(cocktail.thumbnail or "").strip(),
            source=TOOL_NAME_SEARCH_COCKTAILS,
            entity_type=TOOL_RESULT_TYPE_COCKTAIL_RESULTS,
            metadata=metadata.model_dump(exclude_none=True),
            raw_payload=cocktail,
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
    return ToolResult(result=result, evidence_views=evidence_views, hydrated_evidence=hydrated_evidence)




@tool(
    TOOL_NAME_SEARCH_COCKTAILS,
    args_schema=SearchCocktailsArgs,
    description="""
Search for cocktail recipes by name or keyword using TheCocktailDB.

Required fields:
- query (string)

Returns cocktail name, category, alcoholic classification, glass type, ingredients with measures, and instructions.

Example valid call:
{
  "query": "margarita"
}
""",
)
def search_cocktails(query: str) -> ToolResult:
    try:
        response = _cocktail_client.search(query)
        return _tool_result(rerank_cocktail_search_result(response, goal=query, limit=DEFAULT_MEAL_RERANK_LIMIT))
    except RequestException as e:
        return ToolResult.error(f"CocktailDB service unavailable: {e}")
