from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from langchain_core.tools import BaseTool

from agent_runtime.models.plan import PlanStep
from agent_runtime.planning.tool_wrapper import canonical_tool_name


def parse_tool_calls_to_plan_steps(
    tool_calls: Iterable[dict[str, Any]],
    tools: list[BaseTool],
) -> list[PlanStep]:
    """Normalize provider tool calls into executable orchestration steps."""
    available_fields_by_tool = {
        candidate.name: set(getattr(candidate, "available_result_fields", ()))
        for candidate in tools
    }
    steps: list[PlanStep] = []
    for index, call in enumerate(tool_calls):
        if not isinstance(call, dict):
            continue
        name = canonical_tool_name(call.get("name", ""), tools)
        args = call.get("args", {})
        args = args if isinstance(args, dict) else {}

        if "parameters" in args:
            parameters = args.get("parameters", {})
            parameters = parameters if isinstance(parameters, dict) else {}
        else:
            parameters = {
                key: value
                for key, value in args.items()
                if key not in {"hypothesis", "evaluation_fields"}
            }

        hypothesis = str(args.get("hypothesis", ""))
        requested_fields = args.get("evaluation_fields", [])
        evaluation_fields = [
            field.strip()
            for field in requested_fields
            if (
                isinstance(field, str)
                and field.strip()
                and field.strip().lower() not in {"name", "title"}
                and field.strip() in available_fields_by_tool.get(name, set())
            )
        ] if isinstance(requested_fields, list) else []

        metadata = (
            {"evaluation_fields": evaluation_fields}
            if evaluation_fields
            else {}
        )
        steps.append(PlanStep(
            id=str(call.get("id") or f"E{index + 1}"),
            plan=hypothesis,
            hypothesis=hypothesis,
            metadata=metadata,
            tool=name,
            args=parameters,
        ))
    return steps
