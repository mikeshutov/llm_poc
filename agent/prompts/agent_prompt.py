from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from pydantic import BaseModel, Field

from agent.agentstate.model import GeoMetadata
from conversation.models.conversation_models import ConversationContext
from conversation.utils import build_conversation_context_json


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
    geometadata: GeoMetadata | None = None
    task: str = ""
    rules: str = ""
    schema: str = ""
    available_tool_categories: str = ""
    available_tools: str = ""
    previous_iterations: list[PreviousIteration] | None = None
    plan_with_evidence: list[PlanEvidenceStep] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.conversation_context is not None:
            data["conversation_context"] = self.conversation_context.model_dump()
        if self.geometadata is not None:
            data["geometadata"] = self.geometadata.model_dump()
        if self.previous_iterations is not None:
            data["previous_iterations"] = [
                iteration.model_dump() for iteration in self.previous_iterations
            ]
        if self.plan_with_evidence is not None:
            data["plan_with_evidence"] = [
                step.model_dump() for step in self.plan_with_evidence
            ]
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=True, default=str)

    def _serialize_geometadata(self) -> str:
        if self.geometadata is None:
            return ""
        return json.dumps(self.geometadata.model_dump(), indent=2, ensure_ascii=True)

    def _serialize_previous_iterations(self) -> str:
        if not self.previous_iterations:
            return ""
        return json.dumps(
            [iteration.model_dump() for iteration in self.previous_iterations],
            indent=2,
            ensure_ascii=True,
            default=str,
        )

    def _serialize_plan_with_evidence(self) -> str:
        if not self.plan_with_evidence:
            return ""
        return json.dumps(
            [step.model_dump() for step in self.plan_with_evidence],
            indent=2,
            ensure_ascii=True,
            default=str,
        )

    def _append_latest_user_prompt(self, parts: list[str]) -> None:
        parts.extend([
            "Latest User Prompt:",
            self.task,
        ])

    def to_string(self) -> str:
        if self.prompt_kind == "request_analysis":
            parts = [self.instruction.rstrip()]
            if self.conversation_context:
                parts.extend([
                    "Conversation context (JSON):",
                    build_conversation_context_json(self.conversation_context),
                ])
            if self.available_tool_categories:
                parts.extend([
                    "Available categories:",
                    self.available_tool_categories,
                ])
            if self.schema:
                parts.append(f"Response Schema: {self.schema}")
            self._append_latest_user_prompt(parts)
            return "\n\n".join(part for part in parts if part)

        if self.prompt_kind == "planner":
            parts = [self.instruction.rstrip()]
            if self.geometadata:
                parts.extend([
                    "GeoMetadata (JSON):",
                    self._serialize_geometadata(),
                ])
            if self.conversation_context:
                parts.extend([
                    "Conversation Context (JSON):",
                    build_conversation_context_json(self.conversation_context),
                ])
            parts.extend([
                "Allowed Tools:",
                self.available_tools,
            ])
            if self.rules:
                parts.append(self.rules)
            if self.previous_iterations:
                parts.extend([
                    "Previous Iterations (JSON):",
                    self._serialize_previous_iterations(),
                ])
            if self.schema:
                parts.append(self.schema)
            self._append_latest_user_prompt(parts)
            return "\n\n".join(part for part in parts if part)

        if self.prompt_kind == "synthesis":
            parts = [self.instruction.rstrip()]
            if self.rules:
                parts.extend([
                    "Rules:",
                    self.rules,
                ])
            if self.conversation_context:
                parts.extend([
                    "Conversation Context (JSON):",
                    build_conversation_context_json(self.conversation_context),
                ])
            if self.plan_with_evidence:
                parts.extend([
                    "Plan with Evidence (JSON):",
                    self._serialize_plan_with_evidence(),
                ])
            parts.append("Now solve the question or task according to provided evidence above.")
            if self.schema:
                parts.append(self.schema)
            self._append_latest_user_prompt(parts)
            return "\n\n".join(part for part in parts if part)

        raise ValueError(f"Unsupported prompt_kind: {self.prompt_kind}")
