from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from requests.exceptions import RequestException

from integrations.open_er import OpenErClient, ExchangeRates
from request_orchestrator.models.evidence import EvidenceView, HydratedEvidence, ToolResult
from tool.constants import TOOL_NAME_GET_LATEST_EXCHANGE_RATES
from tool.constants import TOOL_RESULT_TYPE_FINANCE

_open_er_client = OpenErClient()


class LatestExchangeRatesArgs(BaseModel):
    base: Optional[str] = Field(
        default="USD",
        description="Base currency code to convert from (e.g. 'USD', 'EUR', 'GBP'). Default: 'USD'.",
    )


class LatestExchangeRateMetadata(BaseModel):
    base: str
    currency: str
    rate: float


def _tool_result(result: ExchangeRates) -> ToolResult:
    hydrated_evidence: list[HydratedEvidence] = []
    evidence_views: list[EvidenceView] = []
    for currency_code, rate in result.rates.items():
        metadata = LatestExchangeRateMetadata(
            base=result.base_code,
            currency=currency_code,
            rate=rate,
        )
        hydrated = HydratedEvidence(
            item_id=currency_code,
            tool_name=TOOL_NAME_GET_LATEST_EXCHANGE_RATES,
            title=f"{result.base_code} to {currency_code}",
            summary=f"Latest rate: 1 {result.base_code} = {rate} {currency_code}.",
            published_at=(result.time_last_update_utc or "").strip(),
            source=TOOL_NAME_GET_LATEST_EXCHANGE_RATES,
            entity_type=TOOL_RESULT_TYPE_FINANCE,
            metadata=metadata.model_dump(exclude_none=True),
            raw_payload={"currency": currency_code, "rate": rate, "exchange_rates": result},
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
    TOOL_NAME_GET_LATEST_EXCHANGE_RATES,
    args_schema=LatestExchangeRatesArgs,
    description="""
Get the latest exchange rates for a base currency against all supported currencies.

Optional fields:
- base: base currency code (default 'USD')

Returns a map of currency codes to their exchange rates relative to the base.

Example valid calls:
{"base": "EUR"}
""",
)
def get_latest_exchange_rates(base: str = "USD") -> ToolResult:
    try:
        return _tool_result(_open_er_client.get_latest(base))
    except RequestException as e:
        return ToolResult.error(f"open.er-api unavailable: {e}")
