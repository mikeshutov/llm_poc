from __future__ import annotations

from typing import Any

from common.utils import normalize_text
from integrations.cocktail_db.models import Cocktail, CocktailSearchResult
from integrations.meal_db.models import Meal, MealSearchResult
from reranker import RerankerCandidate, rerank_candidates


def meal_to_candidate(meal: Meal) -> RerankerCandidate:
    ingredient_names = [cleaned for ingredient in meal.ingredients if (cleaned := normalize_text(ingredient.name))]
    summary_parts = [
        cleaned
        for cleaned in (
            normalize_text(meal.category),
            normalize_text(meal.area),
            normalize_text(meal.tags),
        )
        if cleaned is not None
    ]

    return RerankerCandidate(
        id=meal.id,
        fields={
            "name": normalize_text(meal.name) or meal.name,
            "attributes": ". ".join(summary_parts) if summary_parts else None,
            "instructions": normalize_text(meal.instructions),
            "ingredients": ingredient_names,
        },
    )


def cocktail_to_candidate(cocktail: Cocktail) -> RerankerCandidate:
    ingredient_names = [cleaned for ingredient in cocktail.ingredients if (cleaned := normalize_text(ingredient.name))]
    summary_parts = [
        cleaned
        for cleaned in (
            normalize_text(cocktail.category),
            normalize_text(cocktail.alcoholic),
            normalize_text(cocktail.glass),
            normalize_text(cocktail.tags),
        )
        if cleaned is not None
    ]

    return RerankerCandidate(
        id=cocktail.id,
        fields={
            "name": normalize_text(cocktail.name) or cocktail.name,
            "attributes": ". ".join(summary_parts) if summary_parts else None,
            "instructions": normalize_text(cocktail.instructions),
            "ingredients": ingredient_names,
        },
    )


def rerank_meal_search_result(
    response: MealSearchResult,
    *,
    goal: str | None = None,
    llm: Any | None = None,
    limit: int = 3,
) -> MealSearchResult:
    retrieved_count = len(response.meals)
    if not response.meals:
        return MealSearchResult(meals=[], retrieved_count=0, reranked=True)

    candidates = [meal_to_candidate(meal) for meal in response.meals]
    ranked_candidates = rerank_candidates(candidates, goal=goal, llm=llm, limit=limit)
    meal_by_id = {meal.id: meal for meal in response.meals}
    ranked_meals: list[Meal] = []
    seen_ids: set[str] = set()

    for candidate in ranked_candidates:
        meal = meal_by_id.get(candidate.id)
        if meal is None or meal.id in seen_ids:
            continue
        ranked_meals.append(meal)
        seen_ids.add(meal.id)

    return MealSearchResult(
        meals=ranked_meals[: max(1, limit)],
        retrieved_count=retrieved_count,
        reranked=True,
    )


def rerank_cocktail_search_result(
    response: CocktailSearchResult,
    *,
    goal: str | None = None,
    llm: Any | None = None,
    limit: int = 3,
) -> CocktailSearchResult:
    retrieved_count = len(response.drinks)
    if not response.drinks:
        return CocktailSearchResult(drinks=[], retrieved_count=0, reranked=True)

    candidates = [cocktail_to_candidate(cocktail) for cocktail in response.drinks]
    ranked_candidates = rerank_candidates(candidates, goal=goal, llm=llm, limit=limit)
    cocktail_by_id = {cocktail.id: cocktail for cocktail in response.drinks}
    ranked_cocktails: list[Cocktail] = []
    seen_ids: set[str] = set()

    for candidate in ranked_candidates:
        cocktail = cocktail_by_id.get(candidate.id)
        if cocktail is None or cocktail.id in seen_ids:
            continue
        ranked_cocktails.append(cocktail)
        seen_ids.add(cocktail.id)

    return CocktailSearchResult(
        drinks=ranked_cocktails[: max(1, limit)],
        retrieved_count=retrieved_count,
        reranked=True,
    )
