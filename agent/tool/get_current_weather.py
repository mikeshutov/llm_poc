from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from integrations.open_meteo import OpenMeteoClient
from integrations.open_meteo.models import CurrentWeather, GeocodedLocation

_weather_client = OpenMeteoClient()


class GetCurrentWeatherArgs(BaseModel):
    location: str = Field(
        ...,
        description="City or location name. Example: 'Toronto' or 'Paris, France'.",
    )


class CurrentWeatherResult(BaseModel):
    location: GeocodedLocation
    weather: CurrentWeather


@tool(
    "get_current_weather",
    args_schema=GetCurrentWeatherArgs,
    description="""
Get the current weather conditions for a location.

Required fields:
- location (string)

Returns current temperature, wind speed, wind direction, weather code, and whether it is daytime.

Example valid call:
{
  "location": "Toronto"
}
""",
)
def get_current_weather(location: str) -> Optional[CurrentWeatherResult]:
    result = _weather_client.get_current_for_location(location)
    if result is None:
        return None
    loc, weather = result
    return CurrentWeatherResult(location=loc, weather=weather)
