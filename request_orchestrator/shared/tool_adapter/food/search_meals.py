from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from requests.exceptions import RequestException

from integrations.meal_db import MealDbClient, MealSearchResult
from request_orchestrator.shared.tool_adapter.food.candidate_mapper import rerank_meal_search_result
from request_orchestrator.shared.tool_adapter.food.constants import DEFAULT_MEAL_RERANK_LIMIT

_meal_db_client = MealDbClient()


class SearchMealsArgs(BaseModel):
    query: str = Field(
        ...,
        description="Meal name or keyword to search for. Example: 'pasta', 'chicken', 'sushi'.",
    )


@tool(
    "search_meals",
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
def search_meals(query: str) -> MealSearchResult | str:
    try:
        response = _meal_db_client.search(query)
        return rerank_meal_search_result(response, goal=query, limit=DEFAULT_MEAL_RERANK_LIMIT)
    except RequestException as e:
        return f"MealDB service unavailable: {e}"
