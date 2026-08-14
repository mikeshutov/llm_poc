from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from requests.exceptions import RequestException

from integrations.meal_db import MealDbClient
from integrations.meal_db.models import MealSearchResult
from request_orchestrator.models.evidence import EvidenceUrl, EvidenceView, HydratedEvidence, ToolResult
from request_orchestrator.shared.tool_adapter.food.candidate_mapper import rerank_meal_search_result
from request_orchestrator.shared.tool_adapter.food.constants import DEFAULT_MEAL_RERANK_LIMIT
from tool.constants import TOOL_NAME_SEARCH_MEALS
from tool.constants import TOOL_RESULT_TYPE_MEAL_RESULTS

_meal_db_client = MealDbClient()


class SearchMealsArgs(BaseModel):
    query: str = Field(
        ...,
        description="Meal name or keyword to search for. Example: 'pasta', 'chicken', 'sushi'.",
    )


def _tool_result(result: MealSearchResult) -> ToolResult:
    hydrated_evidence: list[HydratedEvidence] = []
    evidence_views: list[EvidenceView] = []
    for meal in result.meals:
        url = (meal.source or meal.youtube or "").strip()
        urls: list[EvidenceUrl] = []
        if meal.source:
            urls.append(EvidenceUrl(url=meal.source.strip(), url_type="website"))
        if meal.youtube:
            urls.append(EvidenceUrl(url=meal.youtube.strip(), url_type="youtube"))
        summary_parts = [part for part in ((meal.category or "").strip(), (meal.area or "").strip()) if part]
        summary = ". ".join(summary_parts) or (meal.instructions or "").strip() or f"Meal result for {meal.name}."
        hydrated = HydratedEvidence(
            item_id=meal.id,
            tool_name=TOOL_NAME_SEARCH_MEALS,
            title=meal.name,
            summary=summary,
            urls=urls,
            image_url=(meal.thumbnail or "").strip(),
            source=TOOL_NAME_SEARCH_MEALS,
            entity_type=TOOL_RESULT_TYPE_MEAL_RESULTS,
            metadata={
                "category": meal.category,
                "area": meal.area,
                "tags": meal.tags,
                "ingredients": [ingredient.model_dump(exclude_none=True) for ingredient in meal.ingredients],
                "retrieved_count": result.retrieved_count,
                "reranked": result.reranked,
            },
            raw_payload=meal,
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
    TOOL_NAME_SEARCH_MEALS,
    args_schema=SearchMealsArgs,
    description="""
Search for meal recipes by name or keyword using TheMealDB.

Required fields:
- query (string)

Returns meal name, category, cuisine area, ingredients with measures, and cooking instructions.

Example valid call:
{
  "query": "pasta"
}
""",
)
def search_meals(query: str) -> ToolResult:
    try:
        response = _meal_db_client.search(query)
        return _tool_result(rerank_meal_search_result(response, goal=query, limit=DEFAULT_MEAL_RERANK_LIMIT))
    except RequestException as e:
        return ToolResult.error(f"MealDB service unavailable: {e}")
