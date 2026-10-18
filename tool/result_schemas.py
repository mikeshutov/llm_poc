from __future__ import annotations

from pydantic import BaseModel

from request_orchestrator.shared.tool_adapter.food.search_meals import MealEvaluationFields
from tool.constants import TOOL_RESULT_TYPE_MEAL_RESULTS


RESULT_SCHEMAS: dict[str, type[BaseModel]] = {
    TOOL_RESULT_TYPE_MEAL_RESULTS: MealEvaluationFields,
}
