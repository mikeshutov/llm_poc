from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from requests.exceptions import RequestException

from integrations.meal_db import MealDbClient
from integrations.meal_db.models import MealSearchResult
from request_orchestrator.models.evidence import EvidenceUrl, EvidenceUrlType, EvidenceView, ToolMetadata, ToolResult
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


class MealSearchMetadata(BaseModel):
    category: str | None = None
    area: str | None = None
    tags: list[str] = Field(default_factory=list)
    ingredients: list["MealIngredientMetadata"] = Field(default_factory=list)


class MealIngredientMetadata(BaseModel):
    name: str = ""
    measure: str | None = None


def _tool_result(result: MealSearchResult) -> ToolResult:
    evidence: list[EvidenceView] = []
    for meal in result.meals:
        url = (meal.source or meal.youtube or "").strip()
        urls: list[EvidenceUrl] = []
        if meal.source:
            urls.append(EvidenceUrl(url=meal.source.strip(), url_type=EvidenceUrlType.WEBSITE))
        if meal.youtube:
            urls.append(EvidenceUrl(url=meal.youtube.strip(), url_type=EvidenceUrlType.YOUTUBE))
        summary_parts = [part for part in ((meal.category or "").strip(), (meal.area or "").strip()) if part]
        summary = ". ".join(summary_parts) or (meal.instructions or "").strip() or f"Meal result for {meal.name}."
        metadata = MealSearchMetadata(
            category=meal.category,
            area=meal.area,
            tags=[tag.strip() for tag in (meal.tags or "").split(",") if tag.strip()],
            ingredients=[
                MealIngredientMetadata.model_validate(ingredient.model_dump())
                for ingredient in meal.ingredients
            ],
        )
        evidence_view = EvidenceView(
            item_id=meal.id,
            tool_name=TOOL_NAME_SEARCH_MEALS,
            title=meal.name,
            summary=summary,
            urls=urls,
            image_url=(meal.thumbnail or "").strip(),
            source=TOOL_NAME_SEARCH_MEALS,
            entity_type=TOOL_RESULT_TYPE_MEAL_RESULTS,
            llm_metadata=metadata.model_dump(mode="json", exclude_none=True),
            raw_payload=meal,
        )
        evidence.append(evidence_view)
    return ToolResult(
        result=result,
        tool_metadata=ToolMetadata(
            retrieved_count=result.retrieved_count,
            reranked=result.reranked,
        ),

        evidence=evidence,
    )




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
