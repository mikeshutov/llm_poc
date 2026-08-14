from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any

from pydantic import BaseModel, Field
import tiktoken

from common.config import CHUNK_ENCODING
from common.data import is_meaningful_prompt_value, prune_empty_prompt_values
from personalization.profile.models import UserProfile
from conversation.models.conversation_models import ConversationContext
from request_orchestrator.models.evidence import EvidenceView


class PreviousIterationStep(BaseModel):
    step_id: str
    plan: str
    tool: str
    args: dict[str, Any]


class PreviousIteration(BaseModel):
    iteration: int
    has_plan: bool
    steps: list[PreviousIterationStep] = Field(default_factory=list)


class EvidenceStep(BaseModel):
    type: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    evidence: list[EvidenceView] = Field(default_factory=list)


class PromptSectionKeys:
    USER_PROFILE = "user_profile"
    CONVERSATION_CONTEXT = "conversation_context"
    AVAILABLE_TOOL_CATEGORIES = "available_tool_categories"
    AVAILABLE_TOOLS = "available_tools"
    RULES = "rules"
    PREVIOUS_ITERATIONS = "previous_iterations"
    EVIDENCE = "evidence"
    LATEST_USER_PROMPT = "latest_user_prompt"
    TASK = "task"
    SCHEMA = "schema"


@dataclass(frozen=True)
class PromptSectionValue:
    text: str = ""
    token_count: int = 0


@dataclass(frozen=True)
class PromptSection:
    key: str = ""
    heading: str = ""
    value: PromptSectionValue = field(default_factory=PromptSectionValue)
    metadata: dict[str, Any] = field(default_factory=dict)


@lru_cache(maxsize=1)
def _get_prompt_encoding():
    try:
        return tiktoken.get_encoding(CHUNK_ENCODING)
    except Exception:
        return None


def _count_prompt_tokens(text: str) -> int:
    if not text:
        return 0
    encoding = _get_prompt_encoding()
    if encoding is None:
        return max(1, (len(text) + 3) // 4)
    try:
        return len(encoding.encode(text))
    except Exception:
        return max(1, (len(text) + 3) // 4)


@dataclass
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
    evidence: list[EvidenceStep] | None = None
    _sections: dict[str, PromptSection] = field(default_factory=dict, init=False, repr=False)

    def _append_section(
        self,
        heading: str,
        content: str,
        *,
        key: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentPrompt:
        if is_meaningful_prompt_value(content):
            resolved_key = key or heading or f"section_{len(self._sections)}"
            self._sections[resolved_key] = PromptSection(
                key=resolved_key,
                heading=heading,
                value=PromptSectionValue(
                    text=content,
                    token_count=_count_prompt_tokens(content),
                ),
                metadata={} if metadata is None else dict(metadata),
            )
        return self

    def include_user_profile(
        self,
        heading: str = "User Profile (JSON):",
        *,
        key: str = PromptSectionKeys.USER_PROFILE,
        include_management_fields: bool = False,
        include_tone: bool = False,
    ) -> AgentPrompt:
        if self.user_profile is None:
            return self
        return self._append_section(
            heading,
            self._serialize_json(
                self.user_profile.to_prompt_dict(
                    include_management_fields=include_management_fields,
                    include_tone=include_tone,
                )
            ),
            key=key,
            metadata={
                "include_management_fields": include_management_fields,
                "include_tone": include_tone,
            },
        )

    def include_conversation_context(self, heading: str = "Conversation Context (JSON):", *, key: str = PromptSectionKeys.CONVERSATION_CONTEXT) -> AgentPrompt:
        if self.conversation_context is None:
            return self
        from conversation.utils import build_conversation_context_json

        return self._append_section(
            heading,
            build_conversation_context_json(self.conversation_context),
            key=key,
        )

    def include_available_tool_categories(self, heading: str = "Available categories:", *, key: str = PromptSectionKeys.AVAILABLE_TOOL_CATEGORIES) -> AgentPrompt:
        return self._append_section(heading, self.available_tool_categories, key=key)

    def include_available_tools(self, heading: str = "Allowed Tools:", *, key: str = PromptSectionKeys.AVAILABLE_TOOLS) -> AgentPrompt:
        return self._append_section(heading, self.available_tools, key=key)

    def include_rules_section(self, heading: str = "Rules:", *, key: str = PromptSectionKeys.RULES) -> AgentPrompt:
        return self._append_section(heading, self.rules, key=key)

    def include_rules_raw(self, *, key: str = PromptSectionKeys.RULES) -> AgentPrompt:
        return self.include_text(self.rules, key=key)

    def include_previous_iterations(self, heading: str = "Evidence From Previous Plans (JSON):", *, key: str = PromptSectionKeys.PREVIOUS_ITERATIONS) -> AgentPrompt:
        if not self.previous_iterations:
            return self
        return self._append_section(
            heading,
            self._serialize_json(prune_empty_prompt_values([
                iteration.model_dump() for iteration in self.previous_iterations
            ])),
            key=key,
        )

    def include_evidence(
        self,
        heading: str = "Evidence (JSON):",
        *,
        key: str = PromptSectionKeys.EVIDENCE,
    ) -> AgentPrompt:
        if not self.evidence:
            return self
        return self._append_section(
            heading,
            self._serialize_json(self._serialize_evidence_steps()),
            key=key,
        )

    def include_latest_user_prompt(self, heading: str = "Latest User Prompt:", *, key: str = PromptSectionKeys.LATEST_USER_PROMPT) -> AgentPrompt:
        return self._append_section(heading, self.latest_user_prompt or self.task, key=key)

    def include_task(self, heading: str = "Task:", *, key: str = PromptSectionKeys.TASK) -> AgentPrompt:
        return self._append_section(heading, self.task, key=key)

    def include_schema_raw(self, *, key: str = PromptSectionKeys.SCHEMA) -> AgentPrompt:
        return self.include_text(self.schema, key=key)

    def include_schema_as_response_label(self, label: str = "Response Schema:", *, key: str = PromptSectionKeys.SCHEMA) -> AgentPrompt:
        if not self.schema:
            return self
        return self.include_text(f"{label} {self.schema}", key=key)

    def include_text(self, text: str, *, key: str | None = None) -> AgentPrompt:
        if is_meaningful_prompt_value(text):
            resolved_key = key or f"text_{len(self._sections)}"
            self._sections[resolved_key] = PromptSection(
                key=resolved_key,
                value=PromptSectionValue(
                    text=text,
                    token_count=_count_prompt_tokens(text),
                ),
            )
        return self

    def included_sections(self) -> tuple[PromptSection, ...]:
        return tuple(self._sections.values())

    def sections_by_key(self) -> dict[str, PromptSection]:
        return dict(self._sections)

    def get_section(self, key: str) -> PromptSection | None:
        return self._sections.get(key)

    def get_section_content(self, key: str, default: str = "") -> str:
        section = self.get_section(key)
        if section is None:
            return default
        return section.value.text

    def prompt_parts(self) -> tuple[str, ...]:
        parts = [self.instruction.rstrip()]
        for section in self._sections.values():
            if section.heading:
                parts.extend([section.heading, section.value.text])
            else:
                parts.append(section.value.text)
        return tuple(part for part in parts if part)

    def prompt_text(self) -> str:
        return "\n\n".join(self.prompt_parts())

    def prompt_token_count(self) -> int:
        return _count_prompt_tokens(self.prompt_text())

    def to_log_input_object(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt_text(),
            "prompt_token_count": self.prompt_token_count(),
            "prompt_sections": [
                {
                    "key": section.key,
                    "heading": section.heading,
                    "text": section.value.text,
                    "token_count": section.value.token_count,
                }
                for section in self.included_sections()
            ],
        }

    def render(self) -> str:
        return self.prompt_text()

    @staticmethod
    def _serialize_json(value: Any, default: Any = str) -> str:
        return json.dumps(value, indent=2, ensure_ascii=True, default=default)

    def _serialize_evidence_steps(self) -> list[dict[str, Any]]:
        if not self.evidence:
            return []
        serialized_steps: list[dict[str, Any]] = []
        for step in self.evidence:
            serialized_steps.append(
                prune_empty_prompt_values(
                    {
                        "type": step.type,
                        "evidence": [evidence.model_dump() for evidence in step.evidence],
                    }
                )
            )
        return prune_empty_prompt_values(serialized_steps)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("_sections", None)
        if self.conversation_context is not None:
            data[PromptSectionKeys.CONVERSATION_CONTEXT] = prune_empty_prompt_values(
                self.conversation_context.model_dump()
            )
        if self.user_profile is not None:
            user_profile_section = self.get_section(PromptSectionKeys.USER_PROFILE)
            include_management_fields = False
            include_tone = False
            if user_profile_section is not None:
                include_management_fields = bool(user_profile_section.metadata.get("include_management_fields", False))
                include_tone = bool(user_profile_section.metadata.get("include_tone", False))
            data[PromptSectionKeys.USER_PROFILE] = self.user_profile.to_prompt_dict(
                include_management_fields=include_management_fields,
                include_tone=include_tone,
            )
        if self.previous_iterations is not None:
            data[PromptSectionKeys.PREVIOUS_ITERATIONS] = prune_empty_prompt_values([
                iteration.model_dump() for iteration in self.previous_iterations
            ])
        if self.evidence is not None:
            data[PromptSectionKeys.EVIDENCE] = self._serialize_evidence_steps()
        return prune_empty_prompt_values(data)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=True, default=str)
