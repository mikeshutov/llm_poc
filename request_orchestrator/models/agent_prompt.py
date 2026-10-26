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
    available_tools: Any = ""
    evidence: list[EvidenceStep] | None = None
    missing_information: list[str] = field(default_factory=list)
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

    def build_messages(self) -> list[dict[str, Any]]:
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
                # Tool definitions are supplied through bind_tools() by the
                # caller. Do not duplicate the catalog as JSON system text.
                PromptSectionKeys.AVAILABLE_TOOLS,
                PromptSectionKeys.TASK,
            }
        })
        if context_sections:
            system_parts.append("CONTEXT\n" + self._serialize_json(context_sections))

        schema = raw_sections.get(PromptSectionKeys.SCHEMA)
        if isinstance(schema, str) and schema.strip():
            system_parts.append("OUTPUT CONTRACT\n" + schema.strip())

        conversation_context = raw_sections.get(PromptSectionKeys.CONVERSATION_CONTEXT, {})
        history = conversation_context.get("messages", []) if isinstance(conversation_context, dict) else []

        messages: list[dict[str, Any]] = []
        tool_call_ids_by_message_id: dict[str, str] = {}
        emitted_tool_call_ids: set[str] = set()
        orphan_result_payloads: dict[str, dict[str, Any]] = {}
        orphan_result_messages: dict[str, dict[str, Any]] = {}
        tool_result_parent_ids = {
            str(message.get("parent_message_id"))
            for message in history
            if isinstance(message, dict)
            and message.get("kind") == "tool_result"
            and message.get("parent_message_id") is not None
        }
        tool_result_provider_ids = {
            str(message.get("payload", {}).get("provider_tool_call_id"))
            for message in history
            if isinstance(message, dict)
            and message.get("kind") == "tool_result"
            and isinstance(message.get("payload"), dict)
            and message.get("payload", {}).get("provider_tool_call_id")
        }
        if system_parts:
            messages.append({"role": "system", "content": "\n\n".join(system_parts)})

        for message in history:
            if not isinstance(message, dict):
                continue
            kind = message.get("kind")
            visibility = message.get("visibility", "visible")
            role = message.get("role")
            content = message.get("content")
            if visibility == "internal" and kind not in {"tool_call", "tool_result"}:
                continue
            if kind == "tool_result":
                # Tool output is sent as a native tool-role message, but the
                # model-facing result is the normalized evidence projection,
                # not the raw adapter response. Raw output remains available
                # in persistence for replay/debugging.
                payload = message.get("payload", {})
                if not isinstance(payload, dict):
                    payload = {}
                evidence = []
                for item in payload.get("evidence", []):
                    if not isinstance(item, dict):
                        continue
                    evidence.append({
                        key: item.get(key)
                        for key in (
                            "id", "title", "summary",
                            "urls", "image_url", "published_at", "source",
                            "entity_type", "location_name", "llm_metadata",
                        )
                        if item.get(key) not in (None, "", [], {})
                    })
                result_payload = {
                    "tool_name": payload.get("tool_name", ""),
                    "status": payload.get("status", ""),
                    "error_message": payload.get("error_message"),
                    "duration_ms": payload.get("duration_ms"),
                    "evidence": evidence,
                    "tool_metadata": payload.get("tool_metadata", {}),
                }
                tool_content = self._serialize_json(result_payload)
                # LangChain requires every ToolMessage to have a call ID.
                # New records use the parent conversation message ID; older
                # or externally-produced records may only have the provider
                # ID, so retain deterministic fallbacks for replay/context.
                parent_message_id = message.get("parent_message_id")
                matched_tool_call_id = (
                    tool_call_ids_by_message_id.get(str(parent_message_id))
                    if parent_message_id is not None
                    else None
                ) or payload.get("provider_tool_call_id")
                if matched_tool_call_id in emitted_tool_call_ids:
                    messages.append({
                        "role": "tool",
                        "content": tool_content,
                        "tool_call_id": matched_tool_call_id,
                    })
                else:
                    # A bounded/replayed context can contain results without
                    # their originating assistant tool-call messages. Keep
                    # the evidence, but consolidate these historical results
                    # by tool instead of emitting one assistant message per
                    # result.
                    orphan_key = str(payload.get("tool_name") or "unknown_tool")
                    aggregate = orphan_result_payloads.setdefault(
                        orphan_key,
                        {
                            "tool_name": orphan_key,
                            "status": "completed",
                            "error_message": None,
                            "duration_ms": None,
                            "evidence": [],
                            "tool_metadata": {},
                        },
                    )
                    if orphan_key not in orphan_result_messages:
                        orphan_result_messages[orphan_key] = {"role": "assistant", "content": ""}
                        messages.append(orphan_result_messages[orphan_key])
                    if payload.get("status") not in (None, "", "completed"):
                        aggregate["status"] = payload.get("status")
                    if payload.get("error_message"):
                        aggregate["error_message"] = payload.get("error_message")
                    if isinstance(payload.get("duration_ms"), (int, float)):
                        aggregate["duration_ms"] = (
                            (aggregate["duration_ms"] or 0) + payload["duration_ms"]
                        )
                    if isinstance(payload.get("tool_metadata"), dict):
                        aggregate["tool_metadata"].update(payload["tool_metadata"])
                    existing_evidence_ids = {
                        str(item.get("id"))
                        for item in aggregate["evidence"]
                        if isinstance(item, dict) and item.get("id") is not None
                    }
                    for item in result_payload["evidence"]:
                        item_id = str(item.get("id")) if item.get("id") is not None else None
                        if item_id is not None and item_id in existing_evidence_ids:
                            continue
                        aggregate["evidence"].append(item)
            elif kind == "tool_call":
                payload = message.get("payload", {})
                if not isinstance(payload, dict):
                    payload = {}
                # OpenAI requires the following tool result to reference this
                # exact ID. Prefer the provider ID when it was persisted;
                # otherwise use the stable conversation message ID.
                tool_call_id = str(
                    payload.get("provider_tool_call_id")
                    or payload.get("tool_call_id")
                    or message.get("id", "")
                )
                if message.get("id") is not None:
                    tool_call_ids_by_message_id[str(message["id"])] = tool_call_id
                has_result = (
                    str(message.get("id")) in tool_result_parent_ids
                    or tool_call_id in tool_result_provider_ids
                )
                if has_result:
                    emitted_tool_call_ids.add(tool_call_id)
                    messages.append({
                        "role": "assistant",
                        "content": content if isinstance(content, str) else "",
                        "tool_calls": [{
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": str(payload.get("tool_name", "")),
                                "arguments": self._serialize_json(payload.get("input_payload", {}), default=str),
                            },
                        }],
                    })
                else:
                    # Do not emit an orphaned tool-call hypothesis. OpenAI
                    # rejects native calls without responses, and the paired
                    # result/evidence (when present elsewhere in history) is
                    # the useful context to retain.
                    continue
            elif role in {"user", "assistant", "tool"} and isinstance(content, str) and content.strip():
                messages.append({"role": role, "content": content})

        for orphan_key, message in orphan_result_messages.items():
            message["content"] = self._serialize_json(orphan_result_payloads[orphan_key])

        task = raw_sections.get(PromptSectionKeys.TASK)
        if isinstance(task, str) and task.strip():
            messages.append({"role": "user", "content": task.strip()})
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


