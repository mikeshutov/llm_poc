from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from request_orchestrator.models.plan import PlanStep, PlanningResult
from request_orchestrator.shared.planner.tool_wrapper import build_tools, wrap_tool
from tool.models import Tool


class LookupArgs(BaseModel):
    query: str = Field(..., min_length=1)
    limit: int = 5


@tool("lookup", args_schema=LookupArgs)
def lookup(query: str, limit: int = 5) -> dict:
    """Look up matching records."""
    return {"query": query, "limit": limit}


def test_native_tool_schema_wraps_parameters_and_metadata() -> None:
    native = wrap_tool(lookup)

    schema = native.args_schema.model_json_schema()
    assert set(schema["properties"]) == {"parameters", "hypothesis", "metadata", "evaluation_fields"}
    assert set(schema["required"]) == {"parameters", "hypothesis"}
    assert native.invoke({
        "parameters": {"query": "boots", "limit": 2},
        "hypothesis": "Find matching boots.",
        "metadata": {"source": "test"},
    }) == {"query": "boots", "limit": 2}


def test_native_tool_collection_wraps_only_discovered_tools() -> None:
    tools = build_tools([lookup])

    assert [candidate.name for candidate in tools] == ["lookup"]
    assert "parameters" in tools[0].args_schema.model_fields


def test_native_tool_description_exposes_declared_result_fields() -> None:
    native = wrap_tool(Tool(lookup, result_fields=("name", "url", "ingredients")))

    assert "Returns entities with fields: name, url, ingredients" in native.description


def test_native_tool_call_becomes_plan_step() -> None:
    result = PlanningResult(
        steps=[PlanStep(
            id="call-1",
            plan="Find matching boots.",
            hypothesis="Find matching boots.",
            tool="lookup",
            args={"query": "boots"},
            metadata={"experiment": "native"},
        )],
    )
    assert result.steps[0].args == {"query": "boots"}
    assert result.steps[0].metadata == {"experiment": "native"}
