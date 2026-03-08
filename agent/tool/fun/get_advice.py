from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from requests.exceptions import RequestException

from integrations.advice_slip import AdviceSlipClient, AdviceSlip

_advice_client = AdviceSlipClient()


class GetAdviceArgs(BaseModel):
    query: Optional[str] = Field(
        default=None,
        description="Optional keyword to search for relevant advice. Leave empty for a random slip.",
    )


@tool(
    "get_advice",
    args_schema=GetAdviceArgs,
    description="""
Get a random piece of advice, or search for advice on a specific topic.

Optional fields:
- query (string): keyword to find relevant advice. Omit for a random slip.

Example valid calls:
{}
{"query": "money"}
""",
)
def get_advice(query: str | None = None) -> AdviceSlip | list[AdviceSlip] | str:
    try:
        if query:
            return _advice_client.search(query)
        return _advice_client.random()
    except RequestException as e:
        return f"Advice Slip API unavailable: {e}"
