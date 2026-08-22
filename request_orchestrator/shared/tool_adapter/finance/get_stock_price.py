from __future__ import annotations

import yfinance as yf
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from integrations.yahoo_finance import YAHOO_FINANCE_QUOTE_URL_TEMPLATE
from request_orchestrator.models.evidence import EvidenceUrl, EvidenceView, HydratedEvidence, ToolResult
from tool.constants import TOOL_NAME_GET_STOCK_PRICE
from tool.constants import TOOL_RESULT_TYPE_FINANCE


class GetStockPriceArgs(BaseModel):
    ticker: str = Field(description="The stock ticker symbol e.g. AAPL, TSLA, MSFT")


class StockPrice(BaseModel):
    ticker: str
    current_price: float | None
    previous_close: float | None
    day_high: float | None
    day_low: float | None
    market_cap: float | None


class StockPriceMetadata(BaseModel):
    current_price: float | None = None
    previous_close: float | None = None
    market_cap: float | None = None


def _tool_result(result: StockPrice) -> ToolResult:
    url = YAHOO_FINANCE_QUOTE_URL_TEMPLATE.format(ticker=result.ticker).strip()
    summary = (
        f"{result.ticker} last price {result.current_price}. Previous close {result.previous_close}."
        if result.current_price is not None
        else f"Stock price lookup for {result.ticker}."
    )
    metadata = StockPriceMetadata(
        current_price=result.current_price,
        previous_close=result.previous_close,
        market_cap=result.market_cap,
    )
    hydrated = HydratedEvidence(
        item_id=result.ticker,
        tool_name=TOOL_NAME_GET_STOCK_PRICE,
        title=result.ticker,
        summary=summary,
        urls=[EvidenceUrl(url=url, url_type="website")] if url else [],
        source=TOOL_NAME_GET_STOCK_PRICE,
        entity_type=TOOL_RESULT_TYPE_FINANCE,
        metadata=metadata.model_dump(exclude_none=True),
        raw_payload=result,
    )
    return ToolResult(
        result=result,
        evidence_views=[
            EvidenceView(
                item_id=hydrated.item_id,
                title=hydrated.title,
                summary=hydrated.summary,
                metadata=dict(hydrated.metadata),
            )
        ],
        hydrated_evidence=[hydrated],
    )




@tool(
    TOOL_NAME_GET_STOCK_PRICE,
    args_schema=GetStockPriceArgs,
    description="""
Get the current stock price and basic market data for a given ticker symbol.

Example valid calls:
{"ticker": "AAPL"}
{"ticker": "TSLA"}
{"ticker": "MSFT"}
""",
)
def get_stock_price(ticker: str) -> ToolResult:
    try:
        t = yf.Ticker(ticker.upper())
        info = t.fast_info
        return _tool_result(StockPrice(
            ticker=ticker.upper(),
            current_price=info.last_price,
            previous_close=info.previous_close,
            day_high=info.day_high,
            day_low=info.day_low,
            market_cap=info.market_cap,
        ))
    except Exception as e:
        return ToolResult.error(f"Could not retrieve stock price for {ticker}: {e}")
