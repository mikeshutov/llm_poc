from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from requests.exceptions import RequestException

from integrations.cocktail_db import CocktailDbClient, CocktailSearchResult
from request_orchestrator.shared.tool_adapter.food.candidate_mapper import rerank_cocktail_search_result
from request_orchestrator.shared.tool_adapter.food.constants import DEFAULT_MEAL_RERANK_LIMIT

_cocktail_client = CocktailDbClient()


class SearchCocktailsArgs(BaseModel):
    query: str = Field(
        ...,
        description="Cocktail name or keyword to search for. Example: 'margarita', 'mojito', 'gin'.",
    )


@tool(
    "search_cocktails",
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
def search_cocktails(query: str) -> CocktailSearchResult | str:
    try:
        response = _cocktail_client.search(query)
        return rerank_cocktail_search_result(response, goal=query, limit=DEFAULT_MEAL_RERANK_LIMIT)
    except RequestException as e:
        return f"CocktailDB service unavailable: {e}"
