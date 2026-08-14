from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from requests.exceptions import RequestException

from integrations.coingecko import CoinGeckoClient, CoinMarket
from request_orchestrator.models.evidence import EvidenceUrl, EvidenceView, HydratedEvidence, ToolResult
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


def _tool_result(result: list[CoinMarket]) -> ToolResult:
    hydrated_evidence: list[HydratedEvidence] = []
    evidence_views: list[EvidenceView] = []
    for market in result:
        url = f"https://www.coingecko.com/en/coins/{market.id}".strip()
        price_text = f"{market.current_price} {market.symbol.upper()}".strip() if market.current_price is not None else ""
        change_text = (
            f"24h change {market.price_change_percentage_24h:.2f}%"
            if market.price_change_percentage_24h is not None
            else ""
        )
        summary = ". ".join(part for part in (price_text, change_text) if part) or f"Crypto market data for {market.name}."
        hydrated = HydratedEvidence(
            item_id=market.id,
            tool_name=TOOL_NAME_GET_CRYPTO_MARKETS,
            title=market.name,
            summary=summary,
            urls=[EvidenceUrl(url=url, url_type="website")] if url else [],
            image_url=(market.image or "").strip(),
            source=TOOL_NAME_GET_CRYPTO_MARKETS,
            entity_type=TOOL_RESULT_TYPE_CRYPTO_MARKET,
            metadata={
                "symbol": market.symbol,
                "current_price": market.current_price,
                "market_cap": market.market_cap,
                "market_cap_rank": market.market_cap_rank,
                "total_volume": market.total_volume,
                "high_24h": market.high_24h,
                "low_24h": market.low_24h,
                "price_change_24h": market.price_change_24h,
                "price_change_percentage_24h": market.price_change_percentage_24h,
            },
            raw_payload=market,
        )
        hydrated_evidence.append(hydrated)
        evidence_views.append(
            EvidenceView(
                item_id=hydrated.item_id,
                title=hydrated.title,
                summary=hydrated.summary,
                metadata=dict(hydrated.metadata),
            )
        )
    return ToolResult(result=result, evidence_views=evidence_views, hydrated_evidence=hydrated_evidence)




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
