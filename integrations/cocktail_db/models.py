from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CocktailIngredient(BaseModel):
    name: str
    measure: Optional[str] = None


class Cocktail(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(alias="idDrink")
    name: str = Field(alias="strDrink")
    category: Optional[str] = Field(default=None, alias="strCategory")
    alcoholic: Optional[str] = Field(default=None, alias="strAlcoholic")
    glass: Optional[str] = Field(default=None, alias="strGlass")
    instructions: Optional[str] = Field(default=None, alias="strInstructions")
    thumbnail: Optional[str] = Field(default=None, alias="strDrinkThumb")
    tags: Optional[str] = Field(default=None, alias="strTags")
    ingredients: list[CocktailIngredient] = []

    @model_validator(mode="before")
    @classmethod
    def extract_ingredients(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        ingredients: list[CocktailIngredient] = []
        for i in range(1, 16):
            name = (data.get(f"strIngredient{i}") or "").strip()
            if not name:
                continue
            measure = (data.get(f"strMeasure{i}") or "").strip() or None
            ingredients.append(CocktailIngredient(name=name, measure=measure))
        return {**data, "ingredients": ingredients}


class CocktailSearchResult(BaseModel):
    drinks: list[Cocktail] = []
