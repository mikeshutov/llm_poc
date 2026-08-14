from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from integrations.rest_countries import RestCountriesClient, Country
from request_orchestrator.models.evidence import EvidenceView, HydratedEvidence, ToolResult
from tool.constants import TOOL_NAME_COUNTRY_LOOKUP
from tool.constants import TOOL_RESULT_TYPE_COUNTRY

_client = RestCountriesClient()


class CountryLookupArgs(BaseModel):
    country_name: str = Field(
        ...,
        description="The name of the country to look up. Can be a full name or partial name.",
    )


def _tool_result(result: list[Country]) -> ToolResult:
    hydrated_evidence: list[HydratedEvidence] = []
    evidence_views: list[EvidenceView] = []
    for country in result:
        capital_text = ", ".join(country.capital)
        summary_parts = [part for part in (capital_text, country.region, country.subregion) if part]
        summary = ". ".join(summary_parts) or f"Country lookup result for {country.common_name}."
        hydrated = HydratedEvidence(
            item_id=country.common_name,
            tool_name=TOOL_NAME_COUNTRY_LOOKUP,
            title=country.common_name,
            summary=summary,
            source=TOOL_NAME_COUNTRY_LOOKUP,
            entity_type=TOOL_RESULT_TYPE_COUNTRY,
            metadata={
                "official_name": country.official_name,
                "capital": list(country.capital),
                "region": country.region,
                "subregion": country.subregion,
                "population": country.population,
                "currencies": dict(country.currencies),
                "languages": dict(country.languages),
                "flag": country.flag,
            },
            raw_payload=country,
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
