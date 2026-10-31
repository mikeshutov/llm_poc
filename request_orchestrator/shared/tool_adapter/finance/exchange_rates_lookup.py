from langchain_core.tools import tool
from pydantic import BaseModel, Field

from integrations.frankfurter import FrankfurterClient
from integrations.frankfurter.models import ExchangeRatesSeries, ExchangeRatesSnapshot
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
    start_date: str | None = Field(
        default=None,
        description="Optional start date in YYYY-MM-DD format for a time series. Use with end_date when needed.",
    )
    end_date: str | None = Field(
        default=None,
        description="Optional end date in YYYY-MM-DD format for a time series.",
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


def _series_tool_result(result: ExchangeRatesSeries) -> ToolResult:
    evidence: list[EvidenceView] = []
    for date_key, day_rates in result.rates.items():
        for currency_code, rate in day_rates.items():
            metadata = ExchangeRateLookupMetadata(
                base=result.base,
                currency=currency_code,
                rate=rate,
                date=date_key,
            )
            evidence.append(
                EvidenceView(
                    item_id=f"{date_key}:{currency_code}",
                    tool_name=TOOL_NAME_EXCHANGE_RATES_LOOKUP,
                    title=f"{result.base} to {currency_code}",
                    summary=f"Exchange rate on {date_key}: 1 {result.base} = {rate} {currency_code}.",
                    published_at=date_key,
                    source=TOOL_NAME_EXCHANGE_RATES_LOOKUP,
                    entity_type=TOOL_RESULT_TYPE_FINANCE,
                    llm_metadata=metadata.model_dump(exclude_none=True),
                    raw_payload={"date": date_key, "currency": currency_code, "rate": rate, "series": result},
                )
            )
    return ToolResult(result=result, evidence=evidence)


@tool(
    TOOL_NAME_EXCHANGE_RATES_LOOKUP,
    args_schema=ExchangeRatesLookupArgs,
    description="""
Look up currency exchange rates from Frankfurter.

Use this for the latest available rates, a specific historical date, or a date range.


Important:
- If date is omitted, this returns the latest available rates.
- If start_date is provided, this returns a time series.
- Use ISO currency codes such as EUR, USD, CAD, GBP.

""",
)
def exchange_rates_lookup(
    base: str = "EUR",
    symbols: list[str] | None = None,
    date: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> ToolResult:
    if date and start_date:
        raise ValueError("Use either date or start_date, not both.")
    if start_date:
        return _series_tool_result(
            _exchange_rates_client.get_time_series(
                start_date=start_date,
                end_date=end_date,
                base=base,
                symbols=symbols,
            )
        )
    if date:
        return _tool_result(_exchange_rates_client.get_historical_rates(date=date, base=base, symbols=symbols))
    return _tool_result(_exchange_rates_client.get_latest_rates(base=base, symbols=symbols))
