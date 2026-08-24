from langchain_core.tools import tool
from pydantic import BaseModel, Field

from integrations.open_meteo import OPEN_METEO_WEBSITE_URL, OpenMeteoClient
from integrations.open_meteo.models import MonthlyWeatherSummary
from request_orchestrator.models.evidence import EvidenceUrl, EvidenceUrlType, EvidenceView, ToolResult
from tool.constants import TOOL_NAME_GET_HISTORICAL_MONTH_WEATHER
from tool.constants import TOOL_RESULT_TYPE_WEATHER

_weather_client = OpenMeteoClient()


class HistoricalMonthWeatherMetadata(BaseModel):
    year: int
    month: int
    temperature_mean: float | None = None
    temperature_max: float | None = None
    temperature_min: float | None = None
    precipitation_sum: float | None = None


def _tool_result(result: MonthlyWeatherSummary | None) -> ToolResult:
    if result is None:
        return ToolResult(result=None, evidence=[])

    location_name = (result.location_name or "").strip()
    summary = f"{location_name} historical weather for {result.year}-{result.month:02d}."
    metadata = HistoricalMonthWeatherMetadata(
        year=result.year,
        month=result.month,
        temperature_mean=result.temperature_mean,
        temperature_max=result.temperature_max,
        temperature_min=result.temperature_min,
        precipitation_sum=result.precipitation_sum,
    )
    evidence_view = EvidenceView(
        item_id=location_name or f"{result.year}-{result.month:02d}",
        tool_name=TOOL_NAME_GET_HISTORICAL_MONTH_WEATHER,
        title="Historical Monthly Weather",
        summary=summary,
        urls=[EvidenceUrl(url=OPEN_METEO_WEBSITE_URL, url_type=EvidenceUrlType.WEBSITE)],
        location_name=location_name,
        source=TOOL_NAME_GET_HISTORICAL_MONTH_WEATHER,
        entity_type=TOOL_RESULT_TYPE_WEATHER,
        llm_metadata=metadata.model_dump(exclude_none=True),
        raw_payload=result,
    )
    return ToolResult(
        result=result,
        evidence=[evidence_view],
    )


class HistoricalMonthWeatherArgs(BaseModel):
    city: str = Field(
        ...,
        description="City name only. Example: 'Toronto'. Do NOT include year or month.",
    )
    year: int = Field(
        ...,
        description="4-digit year. Example: 2024.",
    )
    month: int = Field(
        ...,
        description="Month number from 1 to 12. Example: 2 for February.",
        ge=1,
        le=12,
    )


@tool(
    TOOL_NAME_GET_HISTORICAL_MONTH_WEATHER,
    args_schema=HistoricalMonthWeatherArgs,
    description="""
Get historical weather data for a specific city, year, and month. This is useful for context on the usual weather in a city.

Required fields:
- city (string)
- year (integer)
- month (integer 1-12)

Do NOT combine year and month into the city field.

Example valid call:
{
  "city": "Toronto",
  "year": 2024,
  "month": 2
}
""",
)
def get_historical_month_weather(city: str, year: int, month: int) -> ToolResult:
    result = _weather_client.get_historical_month(city, year, month)
    return _tool_result(result.model_copy(update={"daily": None}))
