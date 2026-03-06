from langchain_core.tools import tool
from pydantic import BaseModel, Field

from integrations.nager import NagerDateClient
from integrations.nager.models import PublicHoliday

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


@tool(
    "public_holidays_lookup",
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
def public_holidays_lookup(year: int, country_code: str) -> PublicHolidaysResult:
    holidays = _holiday_client.get_public_holidays(year=year, country_code=country_code)
    return PublicHolidaysResult(
        year=year,
        country_code=country_code.strip().upper(),
        holidays=holidays,
    )
