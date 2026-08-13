from __future__ import annotations

from typing import Any

from common.utils import normalize_text
from integrations.cocktail_db.models import Cocktail, CocktailSearchResult
from integrations.meal_db.models import Meal, MealSearchResult
from reranker import Candidate, rerank_candidates


def meal_to_candidate(meal: Meal) -> Candidate:
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

    return Candidate(
        id=meal.id,
        title=normalize_text(meal.name) or meal.name,
        content={
            "name": normalize_text(meal.name),
            "summary": ". ".join(summary_parts) if summary_parts else None,
            "description": normalize_text(meal.instructions),
            "text": ", ".join(ingredient_names) if ingredient_names else None,
            "url": normalize_text(meal.source) or normalize_text(meal.youtube),
            "image_url": normalize_text(meal.thumbnail),
        },
        attributes={
            "category": normalize_text(meal.category),
            "area": normalize_text(meal.area),
            "tags": normalize_text(meal.tags),
        },
        metadata={
            "source": "meal_db",
        },
    )


def cocktail_to_candidate(cocktail: Cocktail) -> Candidate:
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

    return Candidate(
        id=cocktail.id,
        title=normalize_text(cocktail.name) or cocktail.name,
        content={
            "name": normalize_text(cocktail.name),
            "summary": ". ".join(summary_parts) if summary_parts else None,
            "description": normalize_text(cocktail.instructions),
            "text": ", ".join(ingredient_names) if ingredient_names else None,
            "image_url": normalize_text(cocktail.thumbnail),
        },
        attributes={
            "category": normalize_text(cocktail.category),
            "alcoholic": normalize_text(cocktail.alcoholic),
            "glass": normalize_text(cocktail.glass),
            "tags": normalize_text(cocktail.tags),
        },
        metadata={
            "source": "cocktail_db",
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
