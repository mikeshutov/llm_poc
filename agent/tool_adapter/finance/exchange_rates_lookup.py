from langchain_core.tools import tool
from pydantic import BaseModel, Field

from integrations.frankfurter import FrankfurterClient
from integrations.frankfurter.models import ExchangeRatesSnapshot

_exchange_rates_client = FrankfurterClient()


class ExchangeRatesLookupArgs(BaseModel):
    base: str = Field(
        default="EUR",
        description="3-letter ISO currency code to use as the base currency. Example: 'EUR' or 'USD'.",
        min_length=3,
        max_length=3,
    )
    symbols: list[str] | None = Field(
        default=None,
        description="Optional list of 3-letter ISO currency codes to limit the returned rates. Example: ['USD', 'CAD'].",
    )
    date: str | None = Field(
        default=None,
        description="Optional historical date in YYYY-MM-DD format. Leave empty for the latest available rates.",
    )


@tool(
    "exchange_rates_lookup",
    args_schema=ExchangeRatesLookupArgs,
    description="""
Look up currency exchange rates from Frankfurter.

Use this for the latest available rates or for a specific historical date.

Optional fields:
- base (3-letter ISO currency code)
- symbols (list of 3-letter ISO currency codes)
- date (YYYY-MM-DD string)

Important:
- If date is omitted, this returns the latest available rates.
- Use ISO currency codes such as EUR, USD, CAD, GBP.

Example valid call:
{
  "base": "USD",
  "symbols": ["CAD", "EUR"]
}
""",
)
def exchange_rates_lookup(
    base: str = "EUR",
    symbols: list[str] | None = None,
    date: str | None = None,
) -> ExchangeRatesSnapshot:
    if date:
        return _exchange_rates_client.get_historical_rates(date=date, base=base, symbols=symbols)
    return _exchange_rates_client.get_latest_rates(base=base, symbols=symbols)
