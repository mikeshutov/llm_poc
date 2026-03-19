from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from requests.exceptions import RequestException

from integrations.open_er import OpenErClient, ExchangeRates

_open_er_client = OpenErClient()


class LatestExchangeRatesArgs(BaseModel):
    base: Optional[str] = Field(
        default="USD",
        description="Base currency code to convert from (e.g. 'USD', 'EUR', 'GBP'). Default: 'USD'.",
    )


@tool(
    "get_latest_exchange_rates",
    args_schema=LatestExchangeRatesArgs,
    description="""
Get the latest exchange rates for a base currency against all supported currencies.

Optional fields:
- base: base currency code (default 'USD')

Returns a map of currency codes to their exchange rates relative to the base.

Example valid calls:
{}
{"base": "EUR"}
""",
)
def get_latest_exchange_rates(base: str = "USD") -> ExchangeRates | str:
    try:
        return _open_er_client.get_latest(base)
    except RequestException as e:
        return f"open.er-api unavailable: {e}"
