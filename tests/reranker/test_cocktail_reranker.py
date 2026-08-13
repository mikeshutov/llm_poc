from integrations.cocktail_db.models import CocktailSearchResult
from request_orchestrator.shared.tool_adapter.food.candidate_mapper import (
    cocktail_to_candidate,
    rerank_cocktail_search_result,
)
from request_orchestrator.shared.tool_adapter.food.constants import DEFAULT_MEAL_RERANK_LIMIT
from test_utilities.mock_llm import MockLLM


def test_cocktail_to_candidate_maps_cocktail_fields() -> None:
    response = CocktailSearchResult.model_validate(
        {
            "drinks": [
                {
                    "idDrink": "drink-1",
                    "strDrink": "  Margarita  ",
                    "strCategory": " Ordinary Drink ",
                    "strAlcoholic": " Alcoholic ",
                    "strGlass": " Cocktail glass ",
                    "strInstructions": "  Shake with ice and strain.  ",
                    "strDrinkThumb": " https://example.com/margarita.jpg ",
                    "strTags": " Citrus,IBA ",
                    "strIngredient1": " Tequila ",
                    "strIngredient2": "  Lime Juice  ",
                }
            ]
        }
    )

    candidate = cocktail_to_candidate(response.drinks[0])

    assert candidate.id == "drink-1"
    assert candidate.title == "Margarita"
    assert candidate.content["name"] == "Margarita"
    assert candidate.content["summary"] == "Ordinary Drink. Alcoholic. Cocktail glass. Citrus,IBA"
    assert candidate.content["description"] == "Shake with ice and strain."
    assert candidate.content["text"] == "Tequila, Lime Juice"
    assert candidate.content["image_url"] == "https://example.com/margarita.jpg"
    assert candidate.metadata["source"] == "cocktail_db"


def test_rerank_cocktail_search_result_reorders_and_limits_to_three() -> None:
    llm = MockLLM([
        '{"ranked_ids": ["drink-5", "drink-4", "drink-3", "drink-2", "drink-1"]}'
    ])
    response = CocktailSearchResult.model_validate(
        {
            "drinks": [
                {"idDrink": "drink-1", "strDrink": "Drink One", "strInstructions": "desc 1"},
                {"idDrink": "drink-2", "strDrink": "Drink Two", "strInstructions": "desc 2"},
                {"idDrink": "drink-3", "strDrink": "Drink Three", "strInstructions": "desc 3"},
                {"idDrink": "drink-4", "strDrink": "Drink Four", "strInstructions": "desc 4"},
                {"idDrink": "drink-5", "strDrink": "Drink Five", "strInstructions": "desc 5"},
            ]
        }
    )

    reranked = rerank_cocktail_search_result(response, goal="good tequila cocktails", llm=llm, limit=DEFAULT_MEAL_RERANK_LIMIT)

    assert [cocktail.id for cocktail in reranked.drinks] == ["drink-5", "drink-4", "drink-3"]
    assert reranked.retrieved_count == 5
    assert reranked.reranked is True


def test_rerank_cocktail_search_result_skips_llm_when_result_count_is_at_or_below_limit() -> None:
    llm = MockLLM([
        '{"ranked_ids": ["drink-2", "drink-1"]}'
    ])
    response = CocktailSearchResult.model_validate(
        {
            "drinks": [
                {"idDrink": "drink-1", "strDrink": "Drink One", "strInstructions": "desc 1"},
                {"idDrink": "drink-2", "strDrink": "Drink Two", "strInstructions": "desc 2"},
            ]
        }
    )

    reranked = rerank_cocktail_search_result(response, goal="good tequila cocktails", llm=llm, limit=DEFAULT_MEAL_RERANK_LIMIT)

    assert [cocktail.id for cocktail in reranked.drinks] == ["drink-1", "drink-2"]
    assert reranked.retrieved_count == 2
    assert reranked.reranked is True
    assert llm.last_prompt is None
