from langchain_core.tools import tool
from pydantic import BaseModel, Field

from integrations.nager import NagerDateClient
from integrations.nager.models import PublicHoliday
from request_orchestrator.models.evidence import EvidenceView, ToolResult
from tool.constants import TOOL_NAME_PUBLIC_HOLIDAYS_LOOKUP
from tool.constants import TOOL_RESULT_TYPE_CALENDAR

_holiday_client = NagerDateClient()


class PublicHolidaysLookupArgs(BaseModel):
    year: int = Field(
        ...,
        description="4-digit year for the public holiday calendar. Example: 2026.",
    )
    country_code: str = Field(
        ...,
        description="Two-letter ISO country code. Example: 'CA' or 'US'.",
        min_length=2,
        max_length=2,
    )


class PublicHolidaysResult(BaseModel):
    year: int
    country_code: str
    holidays: list[PublicHoliday]


class PublicHolidayMetadata(BaseModel):
    date: object
    local_name: str | None = None
    counties: list[str] | None = None
    launch_year: int | None = None
    types: list[str] = []


def _tool_result(result: PublicHolidaysResult) -> ToolResult:
    evidence: list[EvidenceView] = []
    for holiday in result.holidays:
        metadata = PublicHolidayMetadata(
            date=holiday.date,
            local_name=holiday.local_name,
            counties=holiday.counties,
            launch_year=holiday.launch_year,
            types=list(holiday.types),
        )
        hydrated = EvidenceView(
            item_id=f"{result.country_code}:{holiday.date}:{holiday.name}",
            tool_name=TOOL_NAME_PUBLIC_HOLIDAYS_LOOKUP,
            title=(holiday.name or "").strip() or "Public Holiday",
            summary=f"{holiday.date} in {result.country_code}",
            published_at=str(holiday.date),
            source=TOOL_NAME_PUBLIC_HOLIDAYS_LOOKUP,
            entity_type=TOOL_RESULT_TYPE_CALENDAR,
            llm_metadata=metadata.model_dump(exclude_none=True),
            raw_payload=holiday,
        )
        evidence.append(hydrated)
    return ToolResult(result=result, evidence=evidence)


@tool(
    TOOL_NAME_PUBLIC_HOLIDAYS_LOOKUP,
    args_schema=PublicHolidaysLookupArgs,
    description="""
Look up official public holidays for a country in a specific year.

Use this when you need a holiday calendar, statutory holiday dates, or to check whether a date falls on a public holiday.

Required fields:
- year (integer)
- country_code (2-letter ISO country code)

Example valid call:
{
  "year": 2026,
  "country_code": "CA"
}
""",
)
def public_holidays_lookup(year: int, country_code: str) -> ToolResult:
    holidays = _holiday_client.get_public_holidays(year=year, country_code=country_code)
    return _tool_result(
        PublicHolidaysResult(
            year=year,
            country_code=country_code.strip().upper(),
            holidays=holidays,
        )
    )
