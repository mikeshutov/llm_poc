from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from requests.exceptions import RequestException

from integrations.binance import BinanceClient, Ticker24hr

_binance_client = BinanceClient()


class CryptoTickerArgs(BaseModel):
    symbol: str = Field(
        ...,
        description="Trading pair symbol. Examples: 'BTCUSDT', 'ETHUSDT', 'SOLUSDT'.",
    )


@tool(
    "get_crypto_ticker",
    args_schema=CryptoTickerArgs,
    description="""
Get 24-hour price ticker data for a cryptocurrency trading pair from Binance.

Required fields:
- symbol (string): trading pair, e.g. 'BTCUSDT', 'ETHUSDT', 'SOLUSDT'

Returns last price, 24h high/low, price change, price change percent, and volume.

Example valid call:
{
  "symbol": "BTCUSDT"
}
""",
)
def get_crypto_ticker(symbol: str) -> Ticker24hr | str:
    try:
        return _binance_client.get_ticker(symbol)
    except RequestException as e:
        return f"Binance API unavailable: {e}"
