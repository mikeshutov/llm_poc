from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from integrations.quotable import QuotableClient
from integrations.quotable.models import Quote
from request_orchestrator.models.evidence import EvidenceView, ToolResult
from tool.constants import TOOL_NAME_GET_QUOTE
from tool.constants import TOOL_RESULT_TYPE_QUOTE

_client = QuotableClient()


class GetQuoteArgs(BaseModel):
    query: Optional[str] = Field(
        default=None,
        description="Optional keyword or topic to search for relevant quotes. Leave empty for a random quote.",
    )


class QuoteMetadata(BaseModel):
    tags: list[str] = Field(default_factory=list)


def _normalize_quotes(result: Quote | list[Quote]) -> list[Quote]:
    return result if isinstance(result, list) else [result]


def _tool_result(result: Quote | list[Quote]) -> ToolResult:
    quotes = _normalize_quotes(result)
    evidence: list[EvidenceView] = []
    for quote in quotes:
        summary = f"\"{quote.content}\""
        metadata = QuoteMetadata(tags=list(quote.tags))
        evidence_view = EvidenceView(
            item_id=f"{quote.author}:{quote.content[:40]}",
            tool_name=TOOL_NAME_GET_QUOTE,
            title=quote.author,
            summary=summary,
            source=TOOL_NAME_GET_QUOTE,
            entity_type=TOOL_RESULT_TYPE_QUOTE,
            llm_metadata=metadata.model_dump(exclude_none=True),
            raw_payload=quote,
        )
        evidence.append(evidence_view)
    return ToolResult(result=result, evidence=evidence)




@tool(
    TOOL_NAME_GET_QUOTE,
    args_schema=GetQuoteArgs,
    description="""
Get a random inspirational quote, or search for quotes on a specific topic or by a keyword.

Optional fields:
- query (string): keyword or topic to search for. Omit for a random quote.

Example valid calls:
{}
{"query": "wisdom"}
{"query": "courage"}
""",
)
def get_quote(query: str | None = None) -> ToolResult:
    try:
        if query:
            return _tool_result(_client.search(query))
        return _tool_result(_client.random())
    except Exception as e:
        return ToolResult.error(f"Quotable API error: {e}")
