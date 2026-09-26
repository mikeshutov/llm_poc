from integrations.meal_db.models import MealSearchResult
from request_orchestrator.shared.tool_adapter.food.candidate_mapper import meal_to_candidate, rerank_meal_search_result
from request_orchestrator.shared.tool_adapter.food.constants import DEFAULT_MEAL_RERANK_LIMIT
from test_utilities.mock_llm import MockLLM


def test_meal_to_candidate_maps_meal_fields() -> None:
    response = MealSearchResult.model_validate(
        {
            "meals": [
                {
                    "idMeal": "meal-1",
                    "strMeal": "  Pasta Primavera  ",
                    "strCategory": " Pasta ",
                    "strArea": " Italian ",
                    "strTags": " Quick,Fresh ",
                    "strInstructions": "  Boil   pasta and toss with vegetables.  ",
                    "strMealThumb": " https://example.com/pasta.jpg ",
                    "strSource": " https://example.com/pasta ",
                    "strIngredient1": " Pasta ",
                    "strIngredient2": "  Tomato  ",
                }
            ]
        }
    )

    candidate = meal_to_candidate(response.meals[0])

    assert candidate.id == "meal-1"
    assert candidate.title == "Pasta Primavera"
    assert candidate.content["name"] == "Pasta Primavera"
    assert candidate.content["summary"] == "Pasta. Italian. Quick,Fresh"
    assert candidate.content["description"] == "Boil pasta and toss with vegetables."
    assert candidate.content["text"] == "Pasta, Tomato"
    assert candidate.content["url"] == "https://example.com/pasta"
    assert candidate.content["image_url"] == "https://example.com/pasta.jpg"
    assert candidate.metadata["source"] == "meal_db"


def test_rerank_meal_search_result_reorders_and_limits_to_three() -> None:
    llm = MockLLM([
        '{"ranked_ids": ["meal-5", "meal-4", "meal-3", "meal-2", "meal-1"]}'
    ])
    response = MealSearchResult.model_validate(
        {
            "meals": [
                {"idMeal": "meal-1", "strMeal": "Meal One", "strInstructions": "desc 1"},
                {"idMeal": "meal-2", "strMeal": "Meal Two", "strInstructions": "desc 2"},
                {"idMeal": "meal-3", "strMeal": "Meal Three", "strInstructions": "desc 3"},
                {"idMeal": "meal-4", "strMeal": "Meal Four", "strInstructions": "desc 4"},
                {"idMeal": "meal-5", "strMeal": "Meal Five", "strInstructions": "desc 5"},
            ]
        }
    )

    reranked = rerank_meal_search_result(response, goal="good pasta recipes", llm=llm, limit=DEFAULT_MEAL_RERANK_LIMIT)

    assert [meal.id for meal in reranked.meals] == ["meal-5", "meal-4", "meal-3"]
    assert reranked.retrieved_count == 5
    assert reranked.reranked is True


def test_rerank_meal_search_result_skips_llm_when_result_count_is_at_or_below_limit() -> None:
    llm = MockLLM([
        '{"ranked_ids": ["meal-2", "meal-1"]}'
    ])
    response = MealSearchResult.model_validate(
        {
            "meals": [
                {"idMeal": "meal-1", "strMeal": "Meal One", "strInstructions": "desc 1"},
                {"idMeal": "meal-2", "strMeal": "Meal Two", "strInstructions": "desc 2"},
            ]
        }
    )

    reranked = rerank_meal_search_result(response, goal="good pasta recipes", llm=llm, limit=DEFAULT_MEAL_RERANK_LIMIT)

    assert [meal.id for meal in reranked.meals] == ["meal-1", "meal-2"]
    assert reranked.retrieved_count == 2
    assert reranked.reranked is True
    assert llm.last_prompt is None
