from __future__ import annotations

import math

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class CalculateArgs(BaseModel):
    expression: str = Field(
        description="A mathematical expression to evaluate. Supports standard arithmetic, and math functions like sqrt, sin, cos, log, pi, e etc."
    )


@tool(
    "calculate",
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
def calculate(expression: str) -> str:
    try:
        allowed = {k: v for k, v in vars(math).items() if not k.startswith("_")}
        allowed["abs"] = abs
        allowed["round"] = round
        result = eval(expression, {"__builtins__": {}}, allowed)  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Could not evaluate expression: {e}"
