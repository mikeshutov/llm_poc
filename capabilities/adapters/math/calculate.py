from __future__ import annotations

import math

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from retrieval.evidence import EvidenceView, ToolResult
from capabilities.registry.constants import TOOL_NAME_CALCULATE
from capabilities.registry.constants import TOOL_RESULT_TYPE_CALCULATION


def _tool_result(result: str) -> ToolResult:
    evidence_view = EvidenceView(
        summary=result,
        source=TOOL_NAME_CALCULATE,
        type=TOOL_RESULT_TYPE_CALCULATION,
        data={
            "item_id": result,
            "title": "Calculation Result",
        })
    return ToolResult(
        result=result,
        evidence=[evidence_view],
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




