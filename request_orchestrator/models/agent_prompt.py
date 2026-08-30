from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, Field
import tiktoken

from common.config import CHUNK_ENCODING
from common.data import is_meaningful_prompt_value, prune_empty_prompt_values
from personalization.profile.models import UserProfile
from conversation.models.conversation_models import ConversationContext
from request_orchestrator.models.evidence import EvidenceView

EVIDENCE_VIEW_COMPACT = "compact"
EVIDENCE_VIEW_EVALUATOR = "evaluator"
EvidenceViewMode: TypeAlias = Literal["compact", "evaluator"]


class EvidenceStep(BaseModel):
    type: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    evidence: list[EvidenceView] = Field(default_factory=list)


class PromptSectionKeys:
    USER_PROFILE = "user_profile"
    CONVERSATION_CONTEXT = "conversation_context"
    AVAILABLE_AGENTS = "available_agents"
    AVAILABLE_TOOL_CATEGORIES = "available_tool_categories"
    AVAILABLE_TOOLS = "available_tools"
    RULES = "rules"
    EVIDENCE = "evidence"
    TASK = "task"
    SCHEMA = "schema"


BUILTIN_SECTION_KEYS = (
    PromptSectionKeys.USER_PROFILE,
    PromptSectionKeys.CONVERSATION_CONTEXT,
    PromptSectionKeys.AVAILABLE_AGENTS,
    PromptSectionKeys.AVAILABLE_TOOL_CATEGORIES,
    PromptSectionKeys.AVAILABLE_TOOLS,
    PromptSectionKeys.RULES,
    PromptSectionKeys.EVIDENCE,
    PromptSectionKeys.TASK,
    PromptSectionKeys.SCHEMA,
)

ROLE_RULES_SECTION_KEYS = (PromptSectionKeys.RULES,)
INPUT_SECTION_KEYS = (
    PromptSectionKeys.USER_PROFILE,
    PromptSectionKeys.CONVERSATION_CONTEXT,
    PromptSectionKeys.AVAILABLE_AGENTS,
    PromptSectionKeys.AVAILABLE_TOOL_CATEGORIES,
    PromptSectionKeys.AVAILABLE_TOOLS,
    PromptSectionKeys.EVIDENCE,
    PromptSectionKeys.TASK,
)
OUTPUT_CONTRACT_SECTION_KEYS = (PromptSectionKeys.SCHEMA,)

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
    instruction: str
    conversation_context: ConversationContext | None = None
    user_profile: UserProfile | None = None
    task: str = ""
    rules: str = ""
    schema: str = ""
    available_agents: Any = ""
    available_tool_categories: Any = ""
    available_tools: Any = ""
    evidence: list[EvidenceStep] | None = None
    evidence_view: EvidenceViewMode = EVIDENCE_VIEW_COMPACT
    _enabled_sections: dict[str, dict[str, Any]] = field(default_factory=dict, init=False, repr=False)

    def include_section(
        self,
        key: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> AgentPrompt:
        if key not in BUILTIN_SECTION_KEYS:
            raise KeyError(f"Unknown prompt section key: {key}")
        self._enabled_sections[key] = {} if metadata is None else dict(metadata)
        return self

    @property
    def sections_raw(self) -> dict[str, Any]:
        sections: dict[str, Any] = {}
        for key, metadata in self._enabled_sections.items():
            value: Any = None
            if key == PromptSectionKeys.USER_PROFILE and self.user_profile is not None:
                value = self.user_profile.to_prompt_dict(
                    include_management_fields=bool(metadata.get("include_management_fields", False)),
                    include_tone=bool(metadata.get("include_tone", False)),
                )
            elif key == PromptSectionKeys.CONVERSATION_CONTEXT and self.conversation_context is not None:
                from conversation.utils import build_conversation_context_json

                value = json.loads(build_conversation_context_json(self.conversation_context))
            elif key == PromptSectionKeys.AVAILABLE_AGENTS:
                value = self._normalize_section_value(self.available_agents)
            elif key == PromptSectionKeys.AVAILABLE_TOOL_CATEGORIES:
                value = self._normalize_section_value(self.available_tool_categories)
            elif key == PromptSectionKeys.AVAILABLE_TOOLS:
                value = self._normalize_section_value(self.available_tools)
            elif key == PromptSectionKeys.RULES:
                value = self.rules
            elif key == PromptSectionKeys.EVIDENCE:
                value = self._serialize_evidence_steps()
            elif key == PromptSectionKeys.TASK:
                value = self.task
            elif key == PromptSectionKeys.SCHEMA:
                value = self.schema.strip()
            if is_meaningful_prompt_value(value):
                sections[key] = value
        return prune_empty_prompt_values(sections)

    def build(self) -> str:
        raw_sections = self.sections_raw
        parts: list[str] = []
        parts.append("ROLE / RULES")
        if is_meaningful_prompt_value(self.instruction):
            parts.append(self.instruction.rstrip())
        for key in ROLE_RULES_SECTION_KEYS:
            value = raw_sections.get(key)
            if isinstance(value, str) and is_meaningful_prompt_value(value):
                parts.append(value)

        input_payload = {
            key: raw_sections[key]
            for key in INPUT_SECTION_KEYS
            if key in raw_sections
        }
        if input_payload:
            parts.extend([
                "INPUT",
                self._serialize_json(input_payload),
            ])

        output_parts: list[str] = []
        for key in OUTPUT_CONTRACT_SECTION_KEYS:
            value = raw_sections.get(key)
            if isinstance(value, str) and value.strip():
                output_parts.append(value.strip())
        if output_parts:
            parts.append("OUTPUT CONTRACT")
            parts.extend(output_parts)
        return "\n\n".join(part for part in parts if part)

    @staticmethod
    def _normalize_section_value(value: Any) -> Any:
        if isinstance(value, BaseModel):
            return prune_empty_prompt_values(value.model_dump())
        if isinstance(value, list):
            return [AgentPrompt._normalize_section_value(item) for item in value]
        if isinstance(value, tuple):
            return [AgentPrompt._normalize_section_value(item) for item in value]
        if isinstance(value, dict):
            return {
                key: AgentPrompt._normalize_section_value(item)
                for key, item in value.items()
            }
        if isinstance(value, str):
            normalized = value.strip()
            if not normalized:
                return ""
            try:
                return json.loads(normalized)
            except Exception:
                return value
        return value

    def prompt_token_count(self) -> int:
        return _count_prompt_tokens(self.build())

    def to_log_input_object(self) -> dict[str, Any]:
        return {
            "prompt": self.build(),
            "prompt_token_count": self.prompt_token_count(),
            "sections_raw": self.sections_raw,
        }

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
                        "metadata": dict(step.metadata),
                        "evidence": [
                            (
                                evidence.to_evaluator_view()
                                if self.evidence_view == EVIDENCE_VIEW_EVALUATOR
                                else evidence.compact_view()
                            )
                            for evidence in step.evidence
                        ],
                    }
                )
            )
        return prune_empty_prompt_values(serialized_steps)


