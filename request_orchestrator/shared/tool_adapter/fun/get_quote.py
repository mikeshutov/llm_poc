from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from integrations.quotable import QuotableClient, Quote

_client = QuotableClient()


class GetQuoteArgs(BaseModel):
    query: Optional[str] = Field(
        default=None,
        description="Optional keyword or topic to search for relevant quotes. Leave empty for a random quote.",
    )


@tool(
    "get_quote",
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
def get_quote(query: str | None = None) -> Quote | list[Quote] | str:
    try:
        if query:
            return _client.search(query)
        return _client.random()
    except Exception as e:
        return f"Quotable API error: {e}"
