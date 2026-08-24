from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from requests.exceptions import RequestException

from integrations.coingecko import COINGECKO_WEBSITE_COIN_URL_TEMPLATE, CoinGeckoClient, CoinMarket
from request_orchestrator.models.evidence import EvidenceUrl, EvidenceUrlType, EvidenceView, ToolResult
from tool.constants import TOOL_NAME_GET_CRYPTO_MARKETS
from tool.constants import TOOL_RESULT_TYPE_CRYPTO_MARKET

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


class CryptoMarketMetadata(BaseModel):
    symbol: str
    current_price: float | None = None
    market_cap: float | None = None
    market_cap_rank: int | None = None
    price_change_percentage_24h: float | None = None


def _tool_result(result: list[CoinMarket]) -> ToolResult:
    evidence: list[EvidenceView] = []
    for market in result:
        url = COINGECKO_WEBSITE_COIN_URL_TEMPLATE.format(coin_id=market.id).strip()
        price_text = f"{market.current_price} {market.symbol.upper()}".strip() if market.current_price is not None else ""
        change_text = (
            f"24h change {market.price_change_percentage_24h:.2f}%"
            if market.price_change_percentage_24h is not None
            else ""
        )
        summary = ". ".join(part for part in (price_text, change_text) if part) or f"Crypto market data for {market.name}."
        metadata = CryptoMarketMetadata(
            symbol=market.symbol,
            current_price=market.current_price,
            market_cap=market.market_cap,
            market_cap_rank=market.market_cap_rank,
            price_change_percentage_24h=market.price_change_percentage_24h,
        )
        evidence_view = EvidenceView(
            item_id=market.id,
            tool_name=TOOL_NAME_GET_CRYPTO_MARKETS,
            title=market.name,
            summary=summary,
            urls=[EvidenceUrl(url=url, url_type=EvidenceUrlType.WEBSITE)] if url else [],
            image_url=(market.image or "").strip(),
            source=TOOL_NAME_GET_CRYPTO_MARKETS,
            entity_type=TOOL_RESULT_TYPE_CRYPTO_MARKET,
            llm_metadata=metadata.model_dump(exclude_none=True),
            raw_payload=market,
        )
        evidence.append(evidence_view)
    return ToolResult(result=result, evidence=evidence)




@tool(
    TOOL_NAME_GET_CRYPTO_MARKETS,
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
def get_crypto_markets(vs_currency: str = "usd", per_page: int = 20) -> ToolResult:
    try:
        return _tool_result(_coingecko_client.get_markets(vs_currency=vs_currency, per_page=per_page))
    except RequestException as e:
        return ToolResult.error(f"CoinGecko API unavailable: {e}")
