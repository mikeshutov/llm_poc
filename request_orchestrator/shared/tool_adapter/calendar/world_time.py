from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from integrations.world_time import WorldTimeClient, WorldTime
from request_orchestrator.models.evidence import EvidenceView, ToolResult
from tool.constants import TOOL_NAME_GET_WORLD_TIME
from tool.constants import TOOL_RESULT_TYPE_TIME

_client = WorldTimeClient()


class GetWorldTimeArgs(BaseModel):
    timezone: str = Field(
        ...,
        description="IANA timezone identifier, e.g. 'Europe/London' or 'America/New_York'.",
    )


class WorldTimeMetadata(BaseModel):
    utc_offset: str
    day_of_week: int
    abbreviation: str


def _tool_result(result: WorldTime) -> ToolResult:
    metadata = WorldTimeMetadata(
        utc_offset=result.utc_offset,
        day_of_week=result.day_of_week,
        abbreviation=result.abbreviation,
    )
    evidence_view = EvidenceView(
        item_id=result.timezone,
        tool_name=TOOL_NAME_GET_WORLD_TIME,
        title=result.timezone,
        summary=f"{result.datetime} ({result.utc_offset}, {result.abbreviation})",
        published_at=result.datetime,
        source=TOOL_NAME_GET_WORLD_TIME,
        entity_type=TOOL_RESULT_TYPE_TIME,
        llm_metadata=metadata.model_dump(exclude_none=True),
        raw_payload=result,
    )
    return ToolResult(
        result=result,
        evidence=[evidence_view],
    )




@tool(
    TOOL_NAME_GET_WORLD_TIME,
    args_schema=GetWorldTimeArgs,
    description="""
Get the current date and time for a given timezone.

Returns the local datetime, UTC offset, day of week, and timezone abbreviation.

Required fields:
- timezone (string): IANA timezone identifier

Example valid calls:
{"timezone": "Europe/London"}
{"timezone": "America/New_York"}
{"timezone": "Asia/Tokyo"}
{"timezone": "Australia/Sydney"}
""",
)
def get_world_time(timezone: str) -> ToolResult:
    try:
        return _tool_result(_client.get_time(timezone))
    except Exception as e:
        return ToolResult.error(f"World Time API error: {e}")
