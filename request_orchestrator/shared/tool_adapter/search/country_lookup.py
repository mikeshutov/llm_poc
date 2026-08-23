from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from integrations.rest_countries import RestCountriesClient, Country
from request_orchestrator.models.evidence import EvidenceView, ToolResult
from tool.constants import TOOL_NAME_COUNTRY_LOOKUP
from tool.constants import TOOL_RESULT_TYPE_COUNTRY

_client = RestCountriesClient()


class CountryLookupArgs(BaseModel):
    country_name: str = Field(
        ...,
        description="The name of the country to look up. Can be a full name or partial name.",
    )


class CountryLookupMetadata(BaseModel):
    official_name: str | None = None
    capital: list[str] = []
    region: str | None = None
    subregion: str | None = None
    population: int | None = None
    currencies: dict[str, object] = {}
    languages: dict[str, str] = {}
    flag: str | None = None


def _tool_result(result: list[Country]) -> ToolResult:
    evidence: list[EvidenceView] = []
    for country in result:
        capital_text = ", ".join(country.capital)
        summary_parts = [part for part in (capital_text, country.region, country.subregion) if part]
        summary = ". ".join(summary_parts) or f"Country lookup result for {country.common_name}."
        metadata = CountryLookupMetadata(
            official_name=country.official_name,
            capital=list(country.capital),
            region=country.region,
            subregion=country.subregion,
            population=country.population,
            currencies=dict(country.currencies),
            languages=dict(country.languages),
            flag=country.flag,
        )
        hydrated = EvidenceView(
            item_id=country.common_name,
            tool_name=TOOL_NAME_COUNTRY_LOOKUP,
            title=country.common_name,
            summary=summary,
            source=TOOL_NAME_COUNTRY_LOOKUP,
            entity_type=TOOL_RESULT_TYPE_COUNTRY,
            llm_metadata=metadata.model_dump(exclude_none=True),
            raw_payload=country,
        )
        evidence.append(hydrated)
    return ToolResult(result=result, evidence=evidence)




@tool(
    TOOL_NAME_COUNTRY_LOOKUP,
    args_schema=CountryLookupArgs,
    description="""
Look up information about one or more countries by name.

Returns details including capital, region, population, currencies, languages, and flag emoji.

Required fields:
- country_name (string): full or partial country name to search

Example valid calls:
{"country_name": "Germany"}
{"country_name": "united states"}
{"country_name": "japan"}
""",
)
def country_lookup(country_name: str) -> ToolResult:
    try:
        return _tool_result(_client.search(country_name))
    except Exception as e:
        return ToolResult.error(f"REST Countries API error: {e}")
