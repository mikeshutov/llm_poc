from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from requests.exceptions import RequestException

from integrations.coingecko import CoinGeckoClient, CoinMarket

_coingecko_client = CoinGeckoClient()


class CryptoMarketsArgs(BaseModel):
    vs_currency: Optional[str] = Field(
        default="usd",
        description="Currency to express prices in. Default: 'usd'.",
    )
    per_page: Optional[int] = Field(
        default=20,
        description="Number of coins to return (max 100). Default: 20.",
    )


@tool(
    "get_crypto_markets",
    args_schema=CryptoMarketsArgs,
    description="""
Get top cryptocurrencies by market cap from CoinGecko, including current price, 24h change, volume, and market cap.

Optional fields:
- vs_currency: target currency (default 'usd')
- per_page: number of results (default 20, max 100)

Example valid calls:
{}
{"vs_currency": "eur", "per_page": 10}
""",
)
def get_crypto_markets(vs_currency: str = "usd", per_page: int = 20) -> list[CoinMarket] | str:
    try:
        return _coingecko_client.get_markets(vs_currency=vs_currency, per_page=per_page)
    except RequestException as e:
        return f"CoinGecko API unavailable: {e}"
