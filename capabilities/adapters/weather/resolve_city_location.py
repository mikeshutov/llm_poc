from integrations.open_meteo import OpenMeteoClient
from integrations.open_meteo.models import GeocodedLocation
from pydantic import BaseModel
from retrieval.evidence import EvidenceView, ToolResult
from capabilities.registry.constants import TOOL_NAME_GET_CURRENT_WEATHER
from capabilities.registry.constants import TOOL_RESULT_TYPE_LOCATION

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
        tool_name=TOOL_NAME_GET_CURRENT_WEATHER,
        title=(result.name or "").strip() or "Resolved City",
        summary=f"{(result.name or '').strip()}, {(result.country or '').strip()}",
        location_name=(result.name or "").strip(),
        source=TOOL_NAME_GET_CURRENT_WEATHER,
        entity_type=TOOL_RESULT_TYPE_LOCATION,
        llm_metadata=metadata.model_dump(exclude_none=True),
        raw_payload=result,
    )
    return ToolResult(
        result=result,
        evidence=[evidence_view],
    )


def resolve_city_location(city: str) -> ToolResult:
    """Resolve a city internally for workflows that need normalized location data."""
    return _tool_result(_weather_client.geocode_city(city))
