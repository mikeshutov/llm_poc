from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel
from requests.exceptions import RequestException

from integrations.ip_api import IpApiClient
from integrations.ip_api.models import IpLocation
from request_orchestrator.models.evidence import EvidenceView, ToolResult
from tool.constants import TOOL_NAME_GET_CALLER_LOCATION
from tool.constants import TOOL_RESULT_TYPE_LOCATION

_ip_api_client = IpApiClient()


class CallerLocationMetadata(BaseModel):
    country: str | None = None
    country_code: str | None = None
    region_name: str | None = None
    lat: float | None = None
    lon: float | None = None
    timezone: str | None = None


def _tool_result(result: IpLocation) -> ToolResult:
    location_name = ", ".join(part for part in [result.city, result.region_name, result.country] if part)
    metadata = CallerLocationMetadata(
        country=result.country,
        country_code=result.country_code,
        region_name=result.region_name,
        lat=result.lat,
        lon=result.lon,
        timezone=result.timezone,
    )
    hydrated = EvidenceView(
        item_id=(result.query or result.city or result.country or "").strip(),
        tool_name=TOOL_NAME_GET_CALLER_LOCATION,
        title=location_name or "Caller Location",
        summary=f"{location_name} ({result.timezone})" if result.timezone and location_name else location_name or "Approximate caller location.",
        location_name=(result.city or "").strip(),
        source=TOOL_NAME_GET_CALLER_LOCATION,
        entity_type=TOOL_RESULT_TYPE_LOCATION,
        llm_metadata=metadata.model_dump(exclude_none=True),
        raw_payload=result,
    )
    return ToolResult(
        result=result,
        evidence=[hydrated],
    )




@tool(
    TOOL_NAME_GET_CALLER_LOCATION,
    description="""
Get the approximate geographic location of the caller based on their IP address.

Returns country, region, city, timezone, latitude, and longitude.

Example valid call:
{}
""",
)
def get_caller_location() -> ToolResult:
    try:
        return _tool_result(_ip_api_client.get_location())
    except RequestException as e:
        return ToolResult.error(f"Location service unavailable: {e}")
