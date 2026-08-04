from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from integrations.world_time import WorldTimeClient, WorldTime

_client = WorldTimeClient()


class GetWorldTimeArgs(BaseModel):
    timezone: str = Field(
        ...,
        description="IANA timezone identifier, e.g. 'Europe/London' or 'America/New_York'.",
    )


@tool(
    "get_world_time",
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
def get_world_time(timezone: str) -> WorldTime | str:
    try:
        return _client.get_time(timezone)
    except Exception as e:
        return f"World Time API error: {e}"
