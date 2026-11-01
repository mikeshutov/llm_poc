from __future__ import annotations

from typing import Any, Iterable

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field, create_model


def _envelope_schema(name: str, parameter_schema: type[BaseModel] | None) -> type[BaseModel]:
    parameter_type: Any = parameter_schema or dict[str, Any]
    return create_model(
        f"{name.title().replace('_', '')}CallArgs",
        parameters=(
            parameter_type,
            Field(
                ...,
                description="Executable arguments for the underlying tool.",
            ),
        ),
        hypothesis=(str, Field(..., min_length=1, description="A factual, testable statement about what the returned data is expected to establish.")),
        evaluation_fields=(list[str], Field(default_factory=list, description="Result fields relevant to evaluating the user's request.")),
    )


def wrap_tool(original: BaseTool) -> StructuredTool:
    args_schema = getattr(original, "args_schema", None)
    envelope_schema = _envelope_schema(original.name, args_schema if isinstance(args_schema, type) else None)

    def invoke_envelope(
        parameters: dict[str, Any] | BaseModel | None = None,
        hypothesis: str = "",
        evaluation_fields: list[str] | None = None,
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


def adapt_tool(original: BaseTool) -> BaseTool:
    """Convert an internal capability into a provider-bindable LangChain tool.

    REACT agents must not receive the planner/evaluation envelope, but custom
    capability ``Tool`` records are not themselves accepted by LangChain's
    ``bind_tools``. This adapter preserves the capability's native argument
    schema and execution behavior without adding hypothesis fields.
    """
    if isinstance(original, BaseTool):
        return original

    args_schema = getattr(original, "args_schema", None)

    def invoke_native(**kwargs: Any) -> Any:
        return original.invoke(kwargs)

    return StructuredTool.from_function(
        func=invoke_native,
        name=original.name,
        description=original.description,
        args_schema=args_schema if isinstance(args_schema, type) else None,
    )


def build_tools(
    tools: Iterable[Any],
    *,
    include_evaluation_wrapper: bool = True,
) -> list[BaseTool]:
    return [
        candidate
        if (getattr(candidate, "metadata", None) or {}).get("agent_delegation")
        else wrap_tool(candidate) if include_evaluation_wrapper else adapt_tool(candidate)
        for candidate in tools
    ]
