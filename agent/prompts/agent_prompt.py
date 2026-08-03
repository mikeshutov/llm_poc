from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from pydantic import BaseModel, Field

from personalization.profile.models import UserProfile
from agent.prompt_constants import (
    PLANNER_PROMPT_KIND,
    REQUEST_ANALYSIS_PROMPT_KIND,
    SYNTHESIS_PROMPT_KIND,
)
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
    user_profile: UserProfile | None = None
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
        if self.user_profile is not None:
            data["user_profile"] = self.user_profile.model_dump()
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

    def _serialize_json(self, value: Any, *, default: Any = str) -> str:
        return json.dumps(value, indent=2, ensure_ascii=True, default=default)

    def _serialize_user_profile(self) -> str:
        if self.user_profile is None:
            return ""
        return self._serialize_json(self.user_profile.model_dump())

    def _serialize_previous_iterations(self) -> str:
        if not self.previous_iterations:
            return ""
        return self._serialize_json(
            [iteration.model_dump() for iteration in self.previous_iterations],
        )

    def _serialize_plan_with_evidence(self) -> str:
        if not self.plan_with_evidence:
            return ""
        return self._serialize_json(
            [step.model_dump() for step in self.plan_with_evidence],
        )

    def _append_section(self, parts: list[str], heading: str, content: str) -> None:
        if content:
            parts.extend([heading, content])

    def _append_user_profile(self, parts: list[str]) -> None:
        self._append_section(parts, "User Profile (JSON):", self._serialize_user_profile())

    def _append_conversation_context(self, parts: list[str], heading: str = "Conversation Context (JSON):") -> None:
        if self.conversation_context is None:
            return
        self._append_section(parts, heading, build_conversation_context_json(self.conversation_context))

    def _append_latest_user_prompt(self, parts: list[str]) -> None:
        self._append_section(parts, "Latest User Prompt:", self.task)

    def _join_parts(self, parts: list[str]) -> str:
        return "\n\n".join(part for part in parts if part)

    def _build_parts(
        self,
        *,
        include_user_profile: bool = False,
        include_conversation_context: bool = False,
        conversation_context_heading: str = "Conversation Context (JSON):",
        include_available_tool_categories: bool = False,
        include_available_tools: bool = False,
        include_rules_section: bool = False,
        include_rules_raw: bool = False,
        include_previous_iterations: bool = False,
        include_plan_with_evidence: bool = False,
        include_latest_user_prompt: bool = False,
        schema_as_response_label: bool = False,
        include_schema_raw: bool = False,
        trailing_note: str = "",
    ) -> list[str]:
        parts = [self.instruction.rstrip()]

        if include_user_profile:
            self._append_user_profile(parts)
        if include_conversation_context:
            self._append_conversation_context(parts, heading=conversation_context_heading)
        if include_available_tool_categories:
            self._append_section(parts, "Available categories:", self.available_tool_categories)
        if include_available_tools:
            self._append_section(parts, "Allowed Tools:", self.available_tools)
        if include_rules_section:
            self._append_section(parts, "Rules:", self.rules)
        if include_rules_raw and self.rules:
            parts.append(self.rules)
        if include_previous_iterations:
            self._append_section(parts, "Previous Iterations (JSON):", self._serialize_previous_iterations())
        if include_plan_with_evidence:
            self._append_section(parts, "Plan with Evidence (JSON):", self._serialize_plan_with_evidence())
        if trailing_note:
            parts.append(trailing_note)
        if schema_as_response_label and self.schema:
            parts.append(f"Response Schema: {self.schema}")
        if include_schema_raw and self.schema:
            parts.append(self.schema)
        if include_latest_user_prompt:
            self._append_latest_user_prompt(parts)

        return parts

    def to_string(self) -> str:
        if self.prompt_kind == REQUEST_ANALYSIS_PROMPT_KIND:
            return self._join_parts(
                self._build_parts(
                    include_user_profile=True,
                    include_conversation_context=True,
                    conversation_context_heading="Conversation context (JSON):",
                    include_available_tool_categories=True,
                    include_latest_user_prompt=True,
                    schema_as_response_label=True,
                )
            )

        if self.prompt_kind == PLANNER_PROMPT_KIND:
            return self._join_parts(
                self._build_parts(
                    include_user_profile=True,
                    include_conversation_context=True,
                    include_available_tools=True,
                    include_rules_raw=True,
                    include_previous_iterations=True,
                    include_latest_user_prompt=True,
                    include_schema_raw=True,
                )
            )

        if self.prompt_kind == SYNTHESIS_PROMPT_KIND:
            return self._join_parts(
                self._build_parts(
                    include_user_profile=True,
                    include_rules_section=True,
                    include_conversation_context=True,
                    include_plan_with_evidence=True,
                    trailing_note="Now solve the question or task according to provided evidence above.",
                    include_latest_user_prompt=True,
                    include_schema_raw=True,
                )
            )

        raise ValueError(f"Unsupported prompt_kind: {self.prompt_kind}")
