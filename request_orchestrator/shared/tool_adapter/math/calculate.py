from __future__ import annotations

import math

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from request_orchestrator.models.evidence import EvidenceView, HydratedEvidence, ToolResult
from tool.constants import TOOL_NAME_CALCULATE
from tool.constants import TOOL_RESULT_TYPE_CALCULATION


def _tool_result(result: str) -> ToolResult:
    hydrated = HydratedEvidence(
        item_id=result,
        tool_name=TOOL_NAME_CALCULATE,
        title="Calculation Result",
        summary=result,
        source=TOOL_NAME_CALCULATE,
        entity_type=TOOL_RESULT_TYPE_CALCULATION,
        raw_payload=result,
    )
    return ToolResult(
        result=result,
        evidence_views=[EvidenceView(item_id=hydrated.item_id, title=hydrated.title, summary=hydrated.summary, metadata={})],
        hydrated_evidence=[hydrated],
    )




class CalculateArgs(BaseModel):
    expression: str = Field(
        description="A mathematical expression to evaluate. Supports standard arithmetic, and math functions like sqrt, sin, cos, log, pi, e etc."
    )


@tool(
    TOOL_NAME_CALCULATE,
    args_schema=CalculateArgs,
    description="""
Evaluate a mathematical expression and return the result.
Supports arithmetic operators (+, -, *, /, **, %) and math functions (sqrt, sin, cos, tan, log, log10, pi, e, abs, ceil, floor, round).

Example valid calls:
{"expression": "2 + 2"}
{"expression": "sqrt(144)"}
{"expression": "sin(pi / 2)"}
{"expression": "log(100, 10)"}
{"expression": "(15 * 8) / 3 + 7"}
""",
)
def calculate(expression: str) -> ToolResult:
    try:
        allowed = {k: v for k, v in vars(math).items() if not k.startswith("_")}
        allowed["abs"] = abs
        allowed["round"] = round
        result = eval(expression, {"__builtins__": {}}, allowed)  # noqa: S307
        return _tool_result(str(result))
    except Exception as e:
        return ToolResult.error(f"Could not evaluate expression: {e}")
