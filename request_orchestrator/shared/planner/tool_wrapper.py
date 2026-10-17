from __future__ import annotations

from typing import Any, Iterable

from langchain_core.tools import BaseTool, StructuredTool, tool
from pydantic import BaseModel, Field, create_model


def _envelope_schema(name: str, parameter_schema: type[BaseModel] | None) -> type[BaseModel]:
    parameter_type: Any = parameter_schema or dict[str, Any]
    return create_model(
        f"{name.title().replace('_', '')}CallArgs",
        parameters=(parameter_type, Field(...)),
        hypothesis=(str, Field(..., min_length=1, description="What this call is expected to establish or produce.")),
        metadata=(dict[str, Any], Field(default_factory=dict, description="Additional call metadata for orchestration.")),
    )


def wrap_tool(original: BaseTool) -> StructuredTool:
    args_schema = getattr(original, "args_schema", None)
    envelope_schema = _envelope_schema(original.name, args_schema if isinstance(args_schema, type) else None)

    def invoke_envelope(
        parameters: dict[str, Any] | BaseModel | None = None,
        hypothesis: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        if isinstance(parameters, BaseModel):
            parameters = parameters.model_dump()
        return original.invoke(parameters or {})

    return StructuredTool.from_function(
        func=invoke_envelope,
        name=original.name,
        description=original.description,
        args_schema=envelope_schema,
        metadata={
            "source_tool": original.name,
            "result_type": getattr(original, "result_type", "generic"),
            "version": getattr(original, "version", 1),
            "rate_limit_key": getattr(original, "rate_limit_key", None),
            "retry_policy": getattr(original, "retry_policy", None),
            "rate_limit_policy": getattr(original, "rate_limit_policy", None),
            "cache_policy": getattr(original, "cache_policy", None),
        },
    )


class DelegationParameters(BaseModel):
    agent_name: str = Field(..., min_length=1)
    task: str = Field(..., min_length=1)


def build_delegation_tool(agent_names: Iterable[str]) -> StructuredTool:
    names = ", ".join(sorted(set(agent_names)))
    envelope_schema = _envelope_schema("delegate_agent", DelegationParameters)

    @tool("delegate_agent", args_schema=envelope_schema)
    def delegate_agent(
        parameters: dict[str, Any],
        hypothesis: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Delegate a self-contained task to another agent."""
        return {"parameters": parameters, "hypothesis": hypothesis, "metadata": metadata or {}}

    delegate_agent.description = (
        "Delegate a self-contained task to an available agent. "
        f"Available agents: {names or 'none'}."
    )
    return delegate_agent


def build_tools(tools: Iterable[BaseTool], agent_names: Iterable[str] = ()) -> list[BaseTool]:
    return [*(wrap_tool(candidate) for candidate in tools), build_delegation_tool(agent_names)]
