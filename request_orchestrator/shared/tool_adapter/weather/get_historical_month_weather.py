from langchain_core.tools import tool
from pydantic import BaseModel, Field

from integrations.open_meteo import OpenMeteoClient
from integrations.open_meteo.models import MonthlyWeatherSummary

_weather_client = OpenMeteoClient()


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
    "get_historical_month_weather",
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
def get_historical_month_weather(city: str, year: int, month: int) -> MonthlyWeatherSummary:
    result = _weather_client.get_historical_month(city, year, month)
    return result.model_copy(update={"daily": None})
