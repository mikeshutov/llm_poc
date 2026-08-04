from langchain_core.tools import tool
from pydantic import BaseModel, Field

from integrations.open_meteo import OpenMeteoClient
from integrations.open_meteo.models import GeocodedLocation

_weather_client = OpenMeteoClient()


class ResolveCityLocationArgs(BaseModel):
    city: str = Field(
        ...,
        description="City name only. Example: 'Toronto'. Do not include month, year, or extra context.",
    )


@tool(
    "resolve_city_location",
    args_schema=ResolveCityLocationArgs,
    description="""
Resolve a city into normalized location metadata for weather-aware shopping decisions.

Required fields:
- city (string)

Example valid call:
{
  "city": "Toronto"
}
""",
)
def resolve_city_location(city: str) -> GeocodedLocation:
    return _weather_client.geocode_city(city)
