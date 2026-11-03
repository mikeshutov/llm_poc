from __future__ import annotations

from datetime import date

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from requests.exceptions import RequestException

from capabilities.registry.constants import TOOL_NAME_GET_WEATHER, TOOL_RESULT_TYPE_WEATHER
from integrations.open_meteo import OPEN_METEO_WEBSITE_URL, OpenMeteoClient
from integrations.open_meteo.models import DailyWeather
from retrieval.evidence import EvidenceUrl, EvidenceUrlType, EvidenceView, ToolResult

_weather_client = OpenMeteoClient()


class ForecastWeatherArgs(BaseModel):
    location: str = Field(..., description="City or location name, such as 'Tokyo'.")
    start_date: date | None = Field(
        default=None,
        description="First forecast date in YYYY-MM-DD format. Defaults to today.",
    )
    end_date: date | None = Field(
        default=None,
        description="Last forecast date in YYYY-MM-DD format. Defaults to six days after start_date.",
    )


def _tool_result(location, daily: DailyWeather) -> ToolResult:
    location_name = (location.name or "").strip()
    summary = f"{location_name} forecast for {daily.date[0]} through {daily.date[-1]}."
    evidence = EvidenceView(
        item_id=location_name,
        tool_name=TOOL_NAME_GET_WEATHER,
        title="Weather Forecast",
        summary=summary,
        urls=[EvidenceUrl(url=OPEN_METEO_WEBSITE_URL, url_type=EvidenceUrlType.WEBSITE)],
        location_name=location_name,
        source=TOOL_NAME_GET_WEATHER,
        entity_type=TOOL_RESULT_TYPE_WEATHER,
        llm_metadata={"daily": daily.model_dump(mode="json"), "country": location.country, "timezone": location.timezone},
        raw_payload={"location": location.model_dump(mode="json"), "daily": daily.model_dump(mode="json")},
    )
    return ToolResult(result={"location": location.model_dump(mode="json"), "daily": daily.model_dump(mode="json")}, evidence=[evidence])


@tool(
    TOOL_NAME_GET_WEATHER,
    args_schema=ForecastWeatherArgs,
    description=(
        "Get a daily weather forecast for a location for up to 16 days. Use this for "
        "future weather or forecast requests, including weekend forecasts."
    ),
)
def get_forecast_weather(
    location: str,
    start_date: date | None = None,
    end_date: date | None = None,
) -> ToolResult:
    try:
        result = _weather_client.get_forecast_for_location(
            location,
            start_date=start_date,
            end_date=end_date,
        )
        if result is None:
            return ToolResult(result=None, evidence=[])
        resolved_location, daily = result
        return _tool_result(resolved_location, daily)
    except RequestException as exc:
        return ToolResult.error(f"Weather service unavailable: {exc}")


__all__ = [
    "get_forecast_weather",
]
