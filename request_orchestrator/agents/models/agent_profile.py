from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tool.tools import TOOL_CATEGORIES


DEFAULT_PLANNER_PROMPT_INSTRUCTION = "You are a planning agent."
DEFAULT_PLANNER_RULES = ""
PROFILE_MANAGEMENT_AGENT_NAME = "profile_management"
DEFAULT_SYNTHESIS_INSTRUCTION = (
    "Solve the following task or problem using the provided evidence. "
)


@dataclass(frozen=True)
class AgentProfile:
    name: str
    allowed_categories: set[str] = field(default_factory=set)
    extra_tools: list[Any] = field(default_factory=list)
    planner_instruction: str = DEFAULT_PLANNER_PROMPT_INSTRUCTION
    planner_rules: str = DEFAULT_PLANNER_RULES
    synthesis_instruction: str = DEFAULT_SYNTHESIS_INSTRUCTION
    persist_tool_calls: bool = True

    def allowed_category_names(self) -> set[str]:
        if self.allowed_categories:
            return set(self.allowed_categories)
        return set(TOOL_CATEGORIES.keys())

    def allowed_tools(self) -> list[Any]:
        tools_by_name: dict[str, Any] = {}
        for category_name in self.allowed_category_names():
            category = TOOL_CATEGORIES.get(category_name)
            if category is None:
                continue
            for tool in category.tools:
                tools_by_name[getattr(tool, 'name')] = tool
        for tool in self.extra_tools:
            tools_by_name[getattr(tool, 'name')] = tool
        return list(tools_by_name.values())

    def allowed_tool_names(self) -> set[str]:
        return {getattr(tool, 'name') for tool in self.allowed_tools()}

    def allowed_tool_categories(self) -> dict[str, Any]:
        return {
            name: TOOL_CATEGORIES[name]
            for name in self.allowed_category_names()
            if name in TOOL_CATEGORIES
        }
