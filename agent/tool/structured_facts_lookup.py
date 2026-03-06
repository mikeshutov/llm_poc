from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from integrations.wikidata import WikidataSparqlClient
from integrations.wikidata.models import SparqlResult

_wikidata_client = WikidataSparqlClient()


class StructuredFactsLookupArgs(BaseModel):
    sparql: str = Field(
        ...,
        description="A SPARQL SELECT query for Wikidata structured facts. Use this for stable entity/property facts, not for general prose search.",
    )


@tool(
    "structured_facts_lookup",
    args_schema=StructuredFactsLookupArgs,
    description="""
Look up structured facts from Wikidata using a SPARQL query.

Use this when you need stable entity/property facts such as:
- countries, cities, capitals
- dates, identifiers, classifications
- other factual relationships that fit structured data

Required fields:
- sparql (string)

Important:
- This is a structured facts tool, not a general web search tool.
- Use a SPARQL SELECT query that returns explicit variables.

Example valid call:
{
  "sparql": "SELECT ?countryLabel WHERE { wd:Q172 rdfs:label ?countryLabel FILTER (LANG(?countryLabel) = 'en') } LIMIT 1"
}
""",
)
def structured_facts_lookup(sparql: str) -> SparqlResult:
    return _wikidata_client.query(sparql)
