from langchain_core.tools import tool
from pydantic import BaseModel, Field

from integrations.frankfurter import FrankfurterClient
from integrations.frankfurter.models import ExchangeRatesSeries

_exchange_rates_client = FrankfurterClient()


class ExchangeRatesTimeSeriesArgs(BaseModel):
    start_date: str = Field(
        ...,
        description="Start date in YYYY-MM-DD format.",
    )
    end_date: str | None = Field(
        default=None,
        description="Optional end date in YYYY-MM-DD format. Leave empty to request from the start date forward.",
    )
    base: str = Field(
        default="EUR",
        description="3-letter ISO currency code to use as the base currency.",
        min_length=3,
        max_length=3,
    )
    symbols: list[str] | None = Field(
        default=None,
        description="Optional list of 3-letter ISO currency codes to limit the returned rates.",
    )


@tool(
    "exchange_rates_time_series",
    args_schema=ExchangeRatesTimeSeriesArgs,
    description="""
Look up a time series of exchange rates across a date range from Frankfurter.

Required fields:
- start_date (YYYY-MM-DD string)

Optional fields:
- end_date (YYYY-MM-DD string)
- base (3-letter ISO currency code)
- symbols (list of 3-letter ISO currency codes)

Use this when the user needs exchange-rate changes over time, not just a single-day snapshot.

Example valid call:
{
  "start_date": "2026-01-01",
  "end_date": "2026-01-07",
  "base": "EUR",
  "symbols": ["USD", "CAD"]
}
""",
)
def exchange_rates_time_series(
    start_date: str,
    end_date: str | None = None,
    base: str = "EUR",
    symbols: list[str] | None = None,
) -> ExchangeRatesSeries:
    return _exchange_rates_client.get_time_series(
        start_date=start_date, end_date=end_date, base=base, symbols=symbols
    )
