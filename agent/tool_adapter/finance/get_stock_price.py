from __future__ import annotations

import yfinance as yf
from langchain_core.tools import tool
from pydantic import BaseModel, Field


class GetStockPriceArgs(BaseModel):
    ticker: str = Field(description="The stock ticker symbol e.g. AAPL, TSLA, MSFT")


class StockPrice(BaseModel):
    ticker: str
    current_price: float | None
    previous_close: float | None
    day_high: float | None
    day_low: float | None
    market_cap: float | None


@tool(
    "get_stock_price",
    args_schema=GetStockPriceArgs,
    description="""
Get the current stock price and basic market data for a given ticker symbol.

Example valid calls:
{"ticker": "AAPL"}
{"ticker": "TSLA"}
{"ticker": "MSFT"}
""",
)
def get_stock_price(ticker: str) -> StockPrice | str:
    try:
        t = yf.Ticker(ticker.upper())
        info = t.fast_info
        return StockPrice(
            ticker=ticker.upper(),
            current_price=info.last_price,
            previous_close=info.previous_close,
            day_high=info.day_high,
            day_low=info.day_low,
            market_cap=info.market_cap,
        )
    except Exception as e:
        return f"Could not retrieve stock price for {ticker}: {e}"
