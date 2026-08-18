from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from llm.conversation_model_config import MAIN_AGENT_MODEL_SCOPE
from request_orchestrator.agent_runner.models.agent_profile import AgentProfile


class UserAgent(BaseModel):
    id: UUID
    user_id: str
    name: str
    description: str = ""
    allowed_categories: list[str] = Field(default_factory=list)
    planner_instruction: str
    planner_rules: str = AgentProfile.__dataclass_fields__["planner_rules"].default
    max_turns: int = AgentProfile.__dataclass_fields__["max_turns"].default
    is_active: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None

    def to_agent_profile(self) -> AgentProfile:
        return AgentProfile(
            name=self.name,
            scope=MAIN_AGENT_MODEL_SCOPE,
            allowed_categories=set(self.allowed_categories),
            planner_instruction=self.planner_instruction,
            planner_rules=self.planner_rules,
            request_analysis_selectable=True,
            max_turns=self.max_turns,
        )
