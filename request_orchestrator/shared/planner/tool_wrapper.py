from __future__ import annotations

from typing import Any, Iterable

from langchain_core.tools import BaseTool, StructuredTool
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
    )


def build_tools(tools: Iterable[BaseTool]) -> list[BaseTool]:
    return [wrap_tool(candidate) for candidate in tools]
