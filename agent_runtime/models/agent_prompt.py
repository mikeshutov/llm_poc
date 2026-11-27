from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
import tiktoken

from common.config import CHUNK_ENCODING, ROLE_ASSISTANT, ROLE_TOOL, ROLE_USER
from common.data import bounded_json_value, is_meaningful_prompt_value, prune_empty_prompt_values
from personalization.profile.models import UserProfile
from conversation.models.conversation_models import ConversationContext
from retrieval.evidence import EvidenceView
from capabilities.discovery.models import AvailableTool

MESSAGE_TYPES_BY_ROLE = {
    ROLE_USER: HumanMessage,
    ROLE_ASSISTANT: AIMessage,
    ROLE_TOOL: ToolMessage,
}


class EvidenceStep(BaseModel):
    type: str
    hypothesis: str = ""
    evaluation_fields: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    evidence: list[EvidenceView] = Field(default_factory=list)
    no_results: bool = False


class PromptSectionKeys:
    USER_PROFILE = "user_profile"
    CONVERSATION_CONTEXT = "conversation_context"
    AVAILABLE_TOOL_CATEGORIES = "available_tool_categories"
    AVAILABLE_TOOLS = "available_tools"
    RULES = "rules"
    EVIDENCE = "evidence"
    MISSING_INFORMATION = "missing_information"
    TASK = "task"
    SCHEMA = "schema"
    EVIDENCE_IDS = "evidence_ids"
    WORKING_MEMORY = "working_memory"


BUILTIN_SECTION_KEYS = (
    PromptSectionKeys.USER_PROFILE,
    PromptSectionKeys.CONVERSATION_CONTEXT,
    PromptSectionKeys.AVAILABLE_TOOL_CATEGORIES,
    PromptSectionKeys.AVAILABLE_TOOLS,
    PromptSectionKeys.RULES,
    PromptSectionKeys.EVIDENCE,
    PromptSectionKeys.MISSING_INFORMATION,
    PromptSectionKeys.TASK,
    PromptSectionKeys.SCHEMA,
    PromptSectionKeys.EVIDENCE_IDS,
    PromptSectionKeys.WORKING_MEMORY,
)

ROLE_RULES_SECTION_KEYS = (PromptSectionKeys.RULES,)
INPUT_SECTION_KEYS = (
    PromptSectionKeys.USER_PROFILE,
    PromptSectionKeys.CONVERSATION_CONTEXT,
    PromptSectionKeys.AVAILABLE_TOOL_CATEGORIES,
    PromptSectionKeys.AVAILABLE_TOOLS,
    PromptSectionKeys.EVIDENCE,
    PromptSectionKeys.MISSING_INFORMATION,
    PromptSectionKeys.TASK,
    PromptSectionKeys.EVIDENCE_IDS,
    PromptSectionKeys.WORKING_MEMORY,
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
    available_tool_categories: Any = ""
    available_tools: list[AvailableTool] = field(default_factory=list)
    evidence: list[EvidenceStep] | None = None
    missing_information: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    working_memory: str = ""
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
                value = self.conversation_context.model_dump(mode="json")
            elif key == PromptSectionKeys.AVAILABLE_TOOL_CATEGORIES:
                value = self._normalize_section_value(self.available_tool_categories)
            elif key == PromptSectionKeys.AVAILABLE_TOOLS:
                value = self._normalize_section_value(self.available_tools)
            elif key == PromptSectionKeys.RULES:
                value = self.rules
            elif key == PromptSectionKeys.EVIDENCE:
                value = self._serialize_evidence_steps()
            elif key == PromptSectionKeys.MISSING_INFORMATION:
                value = self.missing_information
            elif key == PromptSectionKeys.TASK:
                value = self.task
            elif key == PromptSectionKeys.EVIDENCE_IDS:
                value = self.evidence_ids
            elif key == PromptSectionKeys.WORKING_MEMORY:
                value = self.working_memory
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

    def build_messages(self) -> list[Any]:
        """Build a provider-neutral chat message list for model invocation.

        The system message contains the orchestration instructions and all
        non-conversational context. Prior visible conversation messages are
        preserved as native user/assistant messages, followed by the current
        task as the final user message.
        """
        raw_sections = self.sections_raw
        system_parts: list[str] = []
        if is_meaningful_prompt_value(self.instruction):
            system_parts.append(self.instruction.strip())
        rules = raw_sections.get(PromptSectionKeys.RULES)
        if isinstance(rules, str) and rules.strip():
            system_parts.append(rules.strip())

        # Conversation history is represented by the native message list
        # below. Keep rolling summaries in the system message because they
        # are context supplied by the application, not user-authored turns.
        # Derived roundtrip projections and follow-up fields are duplicates
        # of the native message stream.
        context_sections: dict[str, Any] = {}
        summary_sections: dict[str, Any] = {}
        conversation_section = raw_sections.get(PromptSectionKeys.CONVERSATION_CONTEXT)
        if isinstance(conversation_section, dict):
            summary_sections = {
                key: conversation_section[key]
                for key in (
                    "conversation_summary",
                    "latest_conversation_summary",
                )
                if is_meaningful_prompt_value(conversation_section.get(key))
            }
            if summary_sections:
                system_parts.append(
                    "CONVERSATION SUMMARY\n" + self._serialize_json(summary_sections)
                )
        context_sections.update({
            key: raw_sections[key]
            for key in INPUT_SECTION_KEYS
            if key in raw_sections
            and key not in {
                PromptSectionKeys.CONVERSATION_CONTEXT,
                # Current tool results are emitted as native tool messages by
                # the execution loop. Do not duplicate them as system JSON.
                PromptSectionKeys.EVIDENCE,
                # Tool definitions are supplied through bind_tools() by the
                # caller. Do not duplicate the catalog as JSON system text.
                PromptSectionKeys.AVAILABLE_TOOLS,
                PromptSectionKeys.TASK,
                # Working memory is a durable runtime instruction/state block;
                # emit it as an explicit system section below.
                PromptSectionKeys.WORKING_MEMORY,
            }
        })
        if context_sections:
            system_parts.append("CONTEXT\n" + self._serialize_json(context_sections))

        working_memory = raw_sections.get(PromptSectionKeys.WORKING_MEMORY)
        if isinstance(working_memory, str) and working_memory.strip():
            system_parts.append("WORKING MEMORY\n" + working_memory.strip())

        schema = raw_sections.get(PromptSectionKeys.SCHEMA)
        if isinstance(schema, str) and schema.strip():
            system_parts.append("OUTPUT CONTRACT\n" + schema.strip())

        history = (
            self.conversation_context.messages
            if self.conversation_context is not None
            and PromptSectionKeys.CONVERSATION_CONTEXT in self._enabled_sections
            else []
        )
        results_by_call = {
            message.parent_message_id: message
            for message in history
            if message.kind == "tool_result" and message.parent_message_id is not None
        }
        messages: list[Any] = []
        if system_parts:
            messages.append(SystemMessage(content="\n\n".join(system_parts)))

        for message in history:
            if message.kind == "tool_call":
                result = results_by_call.get(message.id)
                if result is None:
                    continue
                call_id = str(message.id)
                messages.append(AIMessage(
                    content=message.content,
                    tool_calls=[{
                        "id": call_id,
                        "type": "tool_call",
                        "name": message.payload["tool_name"],
                        "args": message.payload.get("input_payload", {}),
                    }],
                ))
                payload = result.payload
                messages.append(ToolMessage(
                    content=self._serialize_json({
                        "tool_name": payload.get("tool_name", ""),
                        "status": payload.get("status", ""),
                        "error_message": payload.get("error_message"),
                        "duration_ms": payload.get("duration_ms"),
                        "evidence": payload.get("evidence", []),
                    }),
                    tool_call_id=call_id,
                    name=message.payload["tool_name"],
                ))
            elif message.visibility == "visible" and message.role in {ROLE_USER, ROLE_ASSISTANT} and message.content.strip():
                messages.append(MESSAGE_TYPES_BY_ROLE[message.role](content=message.content))

        task = raw_sections.get(PromptSectionKeys.TASK)
        if isinstance(task, str) and task.strip():
            messages.append(HumanMessage(content=task.strip()))
        return messages

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
                        "evaluation_fields": list(step.evaluation_fields),
                        "metadata": dict(step.metadata),
                        "no_results": step.no_results,
                        "evidence": [evidence.model_dump(mode="json") for evidence in step.evidence],
                    }
                )
            )
        return prune_empty_prompt_values(serialized_steps)


