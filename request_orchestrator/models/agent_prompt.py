from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from pydantic import BaseModel, Field

from common.serialization import prune_empty_prompt_values
from personalization.profile.models import UserProfile
from conversation.models.conversation_models import ConversationContext


class PreviousIterationStep(BaseModel):
    step_id: str
    plan: str
    tool: str
    args: dict[str, Any]
    result: Any = None


class PreviousIteration(BaseModel):
    iteration: int
    has_plan: bool
    steps: list[PreviousIterationStep] = Field(default_factory=list)


class PlanEvidenceStep(BaseModel):
    step_id: str
    plan: str
    tool: str
    args: dict[str, Any]
    evidence: Any = None


@dataclass(frozen=True)
class AgentPrompt:
    prompt_kind: str
    instruction: str
    conversation_context: ConversationContext | None = None
    user_profile: UserProfile | None = None
    task: str = ""
    latest_user_prompt: str = ""
    rules: str = ""
    schema: str = ""
    available_tool_categories: str = ""
    available_tools: str = ""
    previous_iterations: list[PreviousIteration] | None = None
    plan_with_evidence: list[PlanEvidenceStep] | None = None
    include_user_attribute_management_fields: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.conversation_context is not None:
            data["conversation_context"] = prune_empty_prompt_values(
                self.conversation_context.model_dump()
            )
        if self.user_profile is not None:
            data["user_profile"] = self.user_profile.to_prompt_dict(
                include_management_fields=self.include_user_attribute_management_fields,
            )
        if self.previous_iterations is not None:
            data["previous_iterations"] = prune_empty_prompt_values([
                iteration.model_dump() for iteration in self.previous_iterations
            ])
        if self.plan_with_evidence is not None:
            data["plan_with_evidence"] = prune_empty_prompt_values([
                step.model_dump() for step in self.plan_with_evidence
            ])
        return prune_empty_prompt_values(data)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=True, default=str)
