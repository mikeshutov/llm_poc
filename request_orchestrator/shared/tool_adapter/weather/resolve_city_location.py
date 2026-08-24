from langchain_core.tools import tool
from pydantic import BaseModel, Field

from integrations.open_meteo import OpenMeteoClient
from integrations.open_meteo.models import GeocodedLocation
from request_orchestrator.models.evidence import EvidenceView, ToolResult
from tool.constants import TOOL_NAME_RESOLVE_CITY_LOCATION
from tool.constants import TOOL_RESULT_TYPE_LOCATION

_weather_client = OpenMeteoClient()


class ResolveCityLocationMetadata(BaseModel):
    country: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    timezone: str | None = None


def _tool_result(result: GeocodedLocation | None) -> ToolResult:
    if result is None:
        return ToolResult(result=None, evidence=[])

    metadata = ResolveCityLocationMetadata(
        country=result.country,
        latitude=result.latitude,
        longitude=result.longitude,
        timezone=result.timezone,
    )
    evidence_view = EvidenceView(
        item_id=(result.name or "").strip(),
        tool_name=TOOL_NAME_RESOLVE_CITY_LOCATION,
        title=(result.name or "").strip() or "Resolved City",
        summary=f"{(result.name or '').strip()}, {(result.country or '').strip()}",
        location_name=(result.name or "").strip(),
        source=TOOL_NAME_RESOLVE_CITY_LOCATION,
        entity_type=TOOL_RESULT_TYPE_LOCATION,
        llm_metadata=metadata.model_dump(exclude_none=True),
        raw_payload=result,
    )
    return ToolResult(
        result=result,
        evidence=[evidence_view],
    )


class ResolveCityLocationArgs(BaseModel):
    city: str = Field(
        ...,
        description="City name only. Example: 'Toronto'. Do not include month, year, or extra context.",
    )


@tool(
    TOOL_NAME_RESOLVE_CITY_LOCATION,
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
def resolve_city_location(city: str) -> ToolResult:
    return _tool_result(_weather_client.geocode_city(city))
