from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from llm.conversation_model_config import EVALUATOR_STAGE, ModelSelection, PLANNER_STAGE
from capabilities.registry.tools import TOOL_CATEGORIES
from agent_runtime.planning.prompts.planner_rules import DEFAULT_PLANNER_PROMPT_INSTRUCTION

class AgentName(StrEnum):
    REQUEST_ORCHESTRATOR = "request_orchestrator"
    PROFILE_MANAGEMENT = "profile_management"
    DOCUMENT_WRITING = "document_writing"
    RESEARCH = "research"
    CODING = "coding"


DEFAULT_SYNTHESIS_INSTRUCTION = (
    "Solve the following task or problem using the provided evidence. "
)
DEFAULT_MAX_TURNS = 10


class AgentDelegationArgs(BaseModel):
    """Default contract used when an agent does not define a custom one."""

    task: str = Field(
        ...,
        min_length=1,
        description="The complete task to delegate to this agent.",
    )
    evidence_ids: list[str] = Field(
        default_factory=list,
        description="Evidence IDs the delegated agent must use as its source material.",
    )


class AgentKind(StrEnum):
    BUILTIN = "builtin"
    USER_AGENT = "user_agent"


class AgentExecutionStrategy(StrEnum):
    PLANNER_EXECUTOR_EVALUATOR = "planner_executor_evaluator"
    REACT = "react"
    DIRECT_RETURN = "direct_return"

    def required_model_stages(self) -> tuple[str, ...]:
        if self == AgentExecutionStrategy.PLANNER_EXECUTOR_EVALUATOR:
            return (PLANNER_STAGE, EVALUATOR_STAGE)
        if self == AgentExecutionStrategy.REACT:
            return (PLANNER_STAGE,)
        if self == AgentExecutionStrategy.DIRECT_RETURN:
            return (PLANNER_STAGE,)
        raise KeyError(f"Unsupported agent execution strategy: {self}")


@dataclass
class AgentProfile:
    name: str
    scope: str
    version: int = 1
    description: str = ""
    kind: AgentKind = AgentKind.BUILTIN
    execution_strategy: AgentExecutionStrategy = AgentExecutionStrategy.PLANNER_EXECUTOR_EVALUATOR
    allowed_categories: set[str] = field(default_factory=set)
    excluded_categories: set[str] = field(default_factory=set)
    static_tools: list[Any] = field(default_factory=list)
    tools_by_name: dict[str, Any] = field(init=False, default_factory=dict)
    tool_categories: dict[str, Any] = field(init=False, default_factory=dict)
    stage_model_selections: dict[str, ModelSelection] = field(default_factory=dict)
    delegatable: bool = True
    max_turns: int = DEFAULT_MAX_TURNS
    planner_instruction: str = DEFAULT_PLANNER_PROMPT_INSTRUCTION
    synthesis_instruction: str = DEFAULT_SYNTHESIS_INSTRUCTION
    delegation_args_schema: type[BaseModel] = AgentDelegationArgs

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
        for tool in self.static_tools:
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
            return set(self.allowed_categories) - self.excluded_categories
        return set(TOOL_CATEGORIES.keys()) - self.excluded_categories

    @property
    def tools(self) -> list[Any]:
        return list(self.tools_by_name.values())

    @property
    def tool_names(self) -> set[str]:
        return set(self.tools_by_name.keys())
