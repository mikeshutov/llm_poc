from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Ingredient(BaseModel):
    name: str
    measure: Optional[str] = None


class Meal(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(alias="idMeal")
    name: str = Field(alias="strMeal")
    category: Optional[str] = Field(default=None, alias="strCategory")
    area: Optional[str] = Field(default=None, alias="strArea")
    instructions: Optional[str] = Field(default=None, alias="strInstructions")
    thumbnail: Optional[str] = Field(default=None, alias="strMealThumb")
    tags: Optional[str] = Field(default=None, alias="strTags")
    youtube: Optional[str] = Field(default=None, alias="strYoutube")
    source: Optional[str] = Field(default=None, alias="strSource")
    ingredients: list[Ingredient] = []

    @model_validator(mode="before")
    @classmethod
    def extract_ingredients(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        ingredients: list[Ingredient] = []
        for i in range(1, 21):
            name = (data.get(f"strIngredient{i}") or "").strip()
            if not name:
                continue
            measure = (data.get(f"strMeasure{i}") or "").strip() or None
            ingredients.append(Ingredient(name=name, measure=measure))
        data = {**data, "ingredients": ingredients}
        return data


class MealSearchResult(BaseModel):
    meals: list[Meal] = []
