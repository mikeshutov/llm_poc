from __future__ import annotations

from datetime import date

from langchain_core.tools import tool
from pydantic import BaseModel, Field, model_validator

from capabilities.adapters.weather.get_current_weather import get_current_weather
from capabilities.adapters.weather.get_forecast_weather import get_forecast_weather
from capabilities.adapters.weather.get_historical_month_weather import get_historical_month_weather
from capabilities.registry.constants import TOOL_NAME_GET_WEATHER
from retrieval.evidence import ToolResult


class WeatherLookupArgs(BaseModel):
    location: str = Field(..., description="City or location name, such as 'Toronto'.")
    start_date: date | None = Field(
        default=None,
        description="Optional first date in YYYY-MM-DD format. Omit for current weather.",
    )
    end_date: date | None = Field(
        default=None,
        description="Optional last date in YYYY-MM-DD format. Omit to use start_date only.",
    )

    @model_validator(mode="after")
    def validate_dates(self) -> "WeatherLookupArgs":
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("start_date must not be after end_date")
        return self


@tool(
    TOOL_NAME_GET_WEATHER,
    args_schema=WeatherLookupArgs,
    description=(
        "Look up weather for a location. Omit dates for current conditions, use today or "
        "future dates for a forecast, and use a past month for historical weather."
    ),
)
def get_weather(
    location: str,
    start_date: date | None = None,
    end_date: date | None = None,
) -> ToolResult:
    if start_date is None and end_date is None:
        return get_current_weather.invoke({"location": location})

    resolved_start = start_date or end_date
    resolved_end = end_date or start_date
    assert resolved_start is not None and resolved_end is not None
    today = date.today()

    if resolved_end < today:
        if resolved_start.year != resolved_end.year or resolved_start.month != resolved_end.month:
            return ToolResult.error("Historical weather lookup currently supports one calendar month at a time.")
        return get_historical_month_weather.invoke({
            "city": location,
            "year": resolved_start.year,
            "month": resolved_start.month,
        })

    if resolved_start < today:
        return ToolResult.error("A weather date range cannot combine historical and forecast dates.")

    return get_forecast_weather.invoke({
        "location": location,
        "start_date": resolved_start.isoformat(),
        "end_date": resolved_end.isoformat(),
    })


__all__ = ["get_weather"]
