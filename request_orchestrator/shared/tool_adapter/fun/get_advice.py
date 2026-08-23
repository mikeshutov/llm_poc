from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from requests.exceptions import RequestException

from integrations.advice_slip import AdviceSlipClient
from integrations.advice_slip.models import AdviceSlip
from request_orchestrator.models.evidence import EvidenceView, ToolResult
from tool.constants import TOOL_NAME_GET_ADVICE
from tool.constants import TOOL_RESULT_TYPE_ADVICE

_advice_client = AdviceSlipClient()


class GetAdviceArgs(BaseModel):
    query: Optional[str] = Field(
        default=None,
        description="Optional keyword to search for relevant advice. Leave empty for a random slip.",
    )


def _normalize_advice(result: AdviceSlip | list[AdviceSlip]) -> list[AdviceSlip]:
    return result if isinstance(result, list) else [result]


def _tool_result(result: AdviceSlip | list[AdviceSlip]) -> ToolResult:
    advice_items = _normalize_advice(result)
    evidence: list[EvidenceView] = []
    for advice in advice_items:
        hydrated = EvidenceView(
            item_id=str(advice.id),
            tool_name=TOOL_NAME_GET_ADVICE,
            title="Advice Slip",
            summary=advice.advice,
            source=TOOL_NAME_GET_ADVICE,
            entity_type=TOOL_RESULT_TYPE_ADVICE,
            raw_payload=advice,
        )
        evidence.append(hydrated)
    return ToolResult(result=result, evidence=evidence)




@tool(
    TOOL_NAME_GET_ADVICE,
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
def get_advice(query: str | None = None) -> ToolResult:
    try:
        if query:
            return _tool_result(_advice_client.search(query))
        return _tool_result(_advice_client.random())
    except RequestException as e:
        return ToolResult.error(f"Advice Slip API unavailable: {e}")
