from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from integrations.rest_countries import RestCountriesClient, Country

_client = RestCountriesClient()


class CountryLookupArgs(BaseModel):
    country_name: str = Field(
        ...,
        description="The name of the country to look up. Can be a full name or partial name.",
    )


@tool(
    "country_lookup",
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
def country_lookup(country_name: str) -> list[Country] | str:
    try:
        return _client.search(country_name)
    except Exception as e:
        return f"REST Countries API error: {e}"
