from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from integrations.wikidata import WikidataSparqlClient
from integrations.wikidata.models import SparqlResult
from request_orchestrator.models.evidence import EvidenceView, ToolResult
from tool.constants import TOOL_NAME_STRUCTURED_FACTS_LOOKUP
from tool.constants import TOOL_RESULT_TYPE_STRUCTURED_FACTS

_wikidata_client = WikidataSparqlClient()


class StructuredFactsLookupArgs(BaseModel):
    sparql: str = Field(
        ...,
        description="A SPARQL SELECT query for Wikidata structured facts. Use this for stable entity/property facts, not for general prose search.",
    )


class StructuredFactsMetadata(BaseModel):
    vars: list[str] = []


def _binding_value(raw_value: object) -> str:
    if isinstance(raw_value, dict):
        nested = raw_value.get("value")
        return str(nested).strip() if nested is not None else ""
    return str(raw_value).strip() if raw_value is not None else ""


def _binding_summary(binding: dict[str, object], vars_list: list[str]) -> str:
    parts: list[str] = []
    for var_name in vars_list:
        value = _binding_value(binding.get(var_name))
        if value:
            parts.append(f"{var_name}={value}")
    return ", ".join(parts) or "Structured fact result."


def _tool_result(result: SparqlResult) -> ToolResult:
    evidence: list[EvidenceView] = []
    for index, binding in enumerate(result.bindings, start=1):
        item_id = (
            _binding_value(binding.get("qid"))
            or _binding_value(binding.get("itemLabel"))
            or _binding_value(binding.get("item"))
            or str(index)
        )
        title = (
            _binding_value(binding.get("itemLabel"))
            or _binding_value(binding.get("label"))
            or _binding_summary(binding, result.vars[:1])
            or f"Structured fact {index}"
        )
        summary = _binding_summary(binding, result.vars)
        url = _binding_value(binding.get("url"))
        metadata = StructuredFactsMetadata(vars=list(result.vars))
        evidence_view = EvidenceView(
            item_id=item_id,
            tool_name=TOOL_NAME_STRUCTURED_FACTS_LOOKUP,
            title=title,
            summary=summary,
            urls=[{"url": url, "url_type": "website"}] if url else [],
            source=TOOL_NAME_STRUCTURED_FACTS_LOOKUP,
            entity_type=TOOL_RESULT_TYPE_STRUCTURED_FACTS,
            llm_metadata=metadata.model_dump(exclude_none=True),
            raw_payload=binding,
        )
        evidence.append(evidence_view)
    return ToolResult(result=result, evidence=evidence)


@tool(
    TOOL_NAME_STRUCTURED_FACTS_LOOKUP,
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
def structured_facts_lookup(sparql: str) -> ToolResult:
    return _tool_result(_wikidata_client.query(sparql))
