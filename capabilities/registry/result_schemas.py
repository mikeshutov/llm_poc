from __future__ import annotations

from pydantic import BaseModel

from capabilities.adapters.food.search_meals import MealEvaluationFields
from capabilities.registry.constants import TOOL_RESULT_TYPE_MEAL_RESULTS
from capabilities.registry.constants import TOOL_RESULT_TYPE_WEB_PAGE


class WebPageResultFields(BaseModel):
    url: str
    title: str
    description: str
    text: str
    status_code: int
    content_type: str


RESULT_SCHEMAS: dict[str, type[BaseModel]] = {
    TOOL_RESULT_TYPE_MEAL_RESULTS: MealEvaluationFields,
    TOOL_RESULT_TYPE_WEB_PAGE: WebPageResultFields,
}
