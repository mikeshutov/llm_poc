from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from llm.conversation_model_config import EVALUATOR_STAGE, ModelSelection, PLANNER_STAGE
from tool.tools import TOOL_CATEGORIES


DEFAULT_PLANNER_PROMPT_INSTRUCTION = "You are a planning agent."
DEFAULT_PLANNER_RULES = ""
PROFILE_MANAGEMENT_AGENT_NAME = "profile_management"
DEFAULT_SYNTHESIS_INSTRUCTION = (
    "Solve the following task or problem using the provided evidence. "
)
DEFAULT_REQUEST_ANALYSIS_GOAL = ""
DEFAULT_MAX_TURNS = 10


class AgentKind(StrEnum):
    BUILTIN = "builtin"
    USER_AGENT = "user_agent"


class AgentExecutionStrategy(StrEnum):
    PLANNER_EXECUTOR_EVALUATOR = "planner_executor_evaluator"

    def required_model_stages(self) -> tuple[str, ...]:
        if self == AgentExecutionStrategy.PLANNER_EXECUTOR_EVALUATOR:
            return (PLANNER_STAGE, EVALUATOR_STAGE)
        raise KeyError(f"Unsupported agent execution strategy: {self}")


@dataclass
class AgentProfile:
    name: str
    scope: str
    description: str = ""
    kind: AgentKind = AgentKind.BUILTIN
    execution_strategy: AgentExecutionStrategy = AgentExecutionStrategy.PLANNER_EXECUTOR_EVALUATOR
    allowed_categories: set[str] = field(default_factory=set)
    extra_tools: list[Any] = field(default_factory=list)
    tools_by_name: dict[str, Any] = field(init=False, default_factory=dict)
    tool_categories: dict[str, Any] = field(init=False, default_factory=dict)
    stage_model_selections: dict[str, ModelSelection] = field(default_factory=dict)
    request_analysis_selectable: bool = True
    max_turns: int = DEFAULT_MAX_TURNS
    request_analysis_goal: str = DEFAULT_REQUEST_ANALYSIS_GOAL
    planner_instruction: str = DEFAULT_PLANNER_PROMPT_INSTRUCTION
    planner_rules: str = DEFAULT_PLANNER_RULES
    synthesis_instruction: str = DEFAULT_SYNTHESIS_INSTRUCTION

    def __post_init__(self) -> None:
        resolved_categories = {
            name: TOOL_CATEGORIES[name]
            for name in self.allowed_category_names()
            if name in TOOL_CATEGORIES
        }
        tools_by_name: dict[str, Any] = {}
        for category in resolved_categories.values():
            for tool in category.tools:
                tools_by_name[tool.name] = tool
        for tool in self.extra_tools:
            tools_by_name[tool.name] = tool

        self.tool_categories = resolved_categories
        self.tools_by_name = tools_by_name

    def model_selection_for_stage(self, stage: str) -> ModelSelection | None:
        # Built-in profiles use the conversation-level model configuration.
        # Only user agents carry explicit, per-agent model overrides.
        if self.kind != AgentKind.USER_AGENT:
            return None
        selection = self.stage_model_selections.get(stage)
        if selection is not None:
            return selection
        return None

    def allowed_category_names(self) -> set[str]:
        if self.allowed_categories:
            return set(self.allowed_categories)
        return set(TOOL_CATEGORIES.keys())

    @property
    def tools(self) -> list[Any]:
        return list(self.tools_by_name.values())

    @property
    def tool_names(self) -> set[str]:
        return set(self.tools_by_name.keys())
