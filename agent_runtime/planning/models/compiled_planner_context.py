from dataclasses import dataclass, field

from langchain_core.tools import BaseTool


#context of specifically the planner anything we should worry about
@dataclass
class CompiledPlannerContext:
    tools: list[BaseTool]
    compiled_tools: str
    rules: dict[str, list[str]] = field(default_factory=dict)
