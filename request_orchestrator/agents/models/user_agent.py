from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from llm.conversation_model_config import MAIN_AGENT_MODEL_SCOPE, ModelSelection
from request_orchestrator.agent_runner.models.agent_profile import (
    AgentExecutionStrategy,
    AgentKind,
    AgentProfile,
)


class UserAgentModelConfig(BaseModel):
    stage: str
    provider: str
    model: str


class UserAgent(BaseModel):
    id: UUID
    user_id: str
    name: str
    description: str = ""
    execution_strategy: AgentExecutionStrategy = AgentExecutionStrategy.PLANNER_EXECUTOR_EVALUATOR
    allowed_categories: list[str] = Field(default_factory=list)
    planner_instruction: str
    planner_rules: str = AgentProfile.__dataclass_fields__["planner_rules"].default
    max_turns: int = AgentProfile.__dataclass_fields__["max_turns"].default
    is_active: bool = True
    model_configs: list[UserAgentModelConfig] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None

    def to_agent_profile(self) -> AgentProfile:
        return AgentProfile(
            name=self.name,
            scope=MAIN_AGENT_MODEL_SCOPE,
            description=self.description,
            kind=AgentKind.USER_AGENT,
            execution_strategy=self.execution_strategy,
            allowed_categories=set(self.allowed_categories),
            stage_model_selections={
                config.stage: ModelSelection(provider=config.provider, model=config.model)
                for config in self.model_configs
            },
            planner_instruction=self.planner_instruction,
            planner_rules=self.planner_rules,
            request_analysis_selectable=True,
            max_turns=self.max_turns,
        )
