from __future__ import annotations

from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator

from llm.conversation_model_config import MAIN_AGENT_MODEL_SCOPE, ModelSelection
from request_orchestrator.agent_runner.models.agent_profile import AgentExecutionStrategy, AgentKind, AgentProfile


class AgentType(StrEnum):
    SYSTEM = "system"
    USER = "user"


class AgentModelConfig(BaseModel):
    stage: str
    provider: str
    model: str


class Agent(BaseModel):
    id: UUID
    agent_type: AgentType = AgentType.USER
    user_id: str | None = None
    name: str
    description: str = ""
    execution_strategy: AgentExecutionStrategy = AgentExecutionStrategy.PLANNER_EXECUTOR_EVALUATOR
    allowed_categories: list[str] = Field(default_factory=list)
    planner_instruction: str
    planner_rules: str = AgentProfile.__dataclass_fields__["planner_rules"].default
    max_turns: int = AgentProfile.__dataclass_fields__["max_turns"].default
    is_active: bool = True
    model_configs: list[AgentModelConfig] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None

    @model_validator(mode="after")
    def validate_ownership(self) -> "Agent":
        if self.agent_type == AgentType.SYSTEM and self.user_id is not None:
            raise ValueError("system agents cannot have a user_id")
        if self.agent_type == AgentType.USER and not self.user_id:
            raise ValueError("user agents require a user_id")
        return self

    @field_validator("user_id", "name", mode="before")
    @classmethod
    def normalize_text(cls, value: str | None) -> str | None:
        return value.strip() if isinstance(value, str) else value

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
