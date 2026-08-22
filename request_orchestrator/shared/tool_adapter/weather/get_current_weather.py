from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from requests.exceptions import RequestException

from integrations.open_meteo import OPEN_METEO_WEBSITE_URL, OpenMeteoClient
from integrations.open_meteo.models import CurrentWeather, GeocodedLocation
from request_orchestrator.models.evidence import EvidenceUrl, EvidenceUrlType, EvidenceView, HydratedEvidence, ToolResult
from tool.constants import TOOL_NAME_GET_CURRENT_WEATHER
from tool.constants import TOOL_RESULT_TYPE_WEATHER

_weather_client = OpenMeteoClient()


class CurrentWeatherMetadata(BaseModel):
    country: str | None = None
    timezone: str | None = None
    time: str | None = None
    temperature: float | None = None
    windspeed: float | None = None
    weathercode: int | None = None
    is_day: int | None = None


def _tool_result(result) -> ToolResult:
    if result is None:
        return ToolResult(result=None, evidence_views=[], hydrated_evidence=[])

    location = result.location
    weather = result.weather
    location_name = (location.name or "").strip()
    country = (location.country or "").strip()
    summary = f"{weather.temperature} C in {location_name}, {country}, wind {weather.windspeed} km/h, at {weather.time}"
    metadata = CurrentWeatherMetadata(
        country=location.country,
        timezone=location.timezone,
        time=weather.time,
        temperature=weather.temperature,
        windspeed=weather.windspeed,
        weathercode=weather.weathercode,
        is_day=weather.is_day,
    )
    hydrated = HydratedEvidence(
        item_id=location_name,
        tool_name=TOOL_NAME_GET_CURRENT_WEATHER,
        title="Get Current Weather",
        summary=summary,
        urls=[EvidenceUrl(url=OPEN_METEO_WEBSITE_URL, url_type=EvidenceUrlType.WEBSITE)],
        location_name=location_name,
        source=TOOL_NAME_GET_CURRENT_WEATHER,
        entity_type=TOOL_RESULT_TYPE_WEATHER,
        metadata=metadata.model_dump(exclude_none=True),
        raw_payload=result,
    )
    return ToolResult(
        result=result,
        evidence_views=[
            EvidenceView(
                item_id=hydrated.item_id,
                title=hydrated.title,
                summary=hydrated.summary,
                metadata=dict(hydrated.metadata),
            )
        ],
        hydrated_evidence=[hydrated],
    )




class GetCurrentWeatherArgs(BaseModel):
    location: str = Field(
        ...,
        description="City or location name. Example: 'Toronto' or 'Paris, France'.",
    )


class CurrentWeatherResult(BaseModel):
    location: GeocodedLocation
    weather: CurrentWeather


@tool(
    TOOL_NAME_GET_CURRENT_WEATHER,
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
def get_current_weather(location: str) -> ToolResult:
    try:
        result = _weather_client.get_current_for_location(location)
        if result is None:
            return _tool_result(None)
        loc, weather = result
        return _tool_result(CurrentWeatherResult(location=loc, weather=weather))
    except RequestException as e:
        return ToolResult.error(f"Weather service unavailable: {e}")
