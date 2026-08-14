from __future__ import annotations

from langchain_core.tools import tool
from requests.exceptions import RequestException

from integrations.ip_api import IpApiClient
from integrations.ip_api.models import IpLocation
from request_orchestrator.models.evidence import EvidenceView, HydratedEvidence, ToolResult
from tool.constants import TOOL_NAME_GET_CALLER_LOCATION
from tool.constants import TOOL_RESULT_TYPE_LOCATION

_ip_api_client = IpApiClient()


def _tool_result(result: IpLocation) -> ToolResult:
    location_name = ", ".join(part for part in [result.city, result.region_name, result.country] if part)
    hydrated = HydratedEvidence(
        item_id=(result.query or result.city or result.country or "").strip(),
        tool_name=TOOL_NAME_GET_CALLER_LOCATION,
        title=location_name or "Caller Location",
        summary=f"{location_name} ({result.timezone})" if result.timezone and location_name else location_name or "Approximate caller location.",
        location_name=(result.city or "").strip(),
        source=TOOL_NAME_GET_CALLER_LOCATION,
        entity_type=TOOL_RESULT_TYPE_LOCATION,
        metadata={
            "country": result.country,
            "country_code": result.country_code,
            "region": result.region,
            "region_name": result.region_name,
            "zip": result.zip,
            "lat": result.lat,
            "lon": result.lon,
            "timezone": result.timezone,
            "isp": result.isp,
            "org": result.org,
            "query": result.query,
        },
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
