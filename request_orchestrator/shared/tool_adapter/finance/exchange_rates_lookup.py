from langchain_core.tools import tool
from pydantic import BaseModel, Field

from integrations.frankfurter import FrankfurterClient
from integrations.frankfurter.models import ExchangeRatesSnapshot
from request_orchestrator.models.evidence import EvidenceView, ToolResult
from tool.constants import TOOL_NAME_EXCHANGE_RATES_LOOKUP
from tool.constants import TOOL_RESULT_TYPE_FINANCE

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


class ExchangeRateLookupMetadata(BaseModel):
    base: str
    currency: str
    rate: float
    date: str


def _tool_result(result: ExchangeRatesSnapshot) -> ToolResult:
    evidence: list[EvidenceView] = []
    for currency_code, rate in result.rates.items():
        metadata = ExchangeRateLookupMetadata(
            base=result.base,
            currency=currency_code,
            rate=rate,
            date=result.date,
        )
        evidence_view = EvidenceView(
            item_id=currency_code,
            tool_name=TOOL_NAME_EXCHANGE_RATES_LOOKUP,
            title=f"{result.base} to {currency_code}",
            summary=f"Exchange rate on {result.date}: 1 {result.base} = {rate} {currency_code}.",
            published_at=result.date,
            source=TOOL_NAME_EXCHANGE_RATES_LOOKUP,
            entity_type=TOOL_RESULT_TYPE_FINANCE,
            llm_metadata=metadata.model_dump(exclude_none=True),
            raw_payload={"currency": currency_code, "rate": rate, "snapshot": result},
        )
        evidence.append(evidence_view)
    return ToolResult(result=result, evidence=evidence)


@tool(
    TOOL_NAME_EXCHANGE_RATES_LOOKUP,
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
) -> ToolResult:
    if date:
        return _tool_result(_exchange_rates_client.get_historical_rates(date=date, base=base, symbols=symbols))
    return _tool_result(_exchange_rates_client.get_latest_rates(base=base, symbols=symbols))
