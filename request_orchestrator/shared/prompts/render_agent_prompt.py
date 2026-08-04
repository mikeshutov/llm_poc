from __future__ import annotations

import json
from typing import Any

from conversation.utils import build_conversation_context_json
from request_orchestrator.constants import (
    PLANNER_PROMPT_KIND,
    REQUEST_ANALYSIS_PROMPT_KIND,
    SYNTHESIS_PROMPT_KIND,
)
from request_orchestrator.models.agent_prompt import AgentPrompt


def _serialize_json(value: Any, *, default: Any = str) -> str:
    return json.dumps(value, indent=2, ensure_ascii=True, default=default)


def _serialize_user_profile(prompt: AgentPrompt) -> str:
    if prompt.user_profile is None:
        return ""
    return _serialize_json(prompt.user_profile.to_prompt_dict())


def _serialize_previous_iterations(prompt: AgentPrompt) -> str:
    if not prompt.previous_iterations:
        return ""
    return _serialize_json([iteration.model_dump() for iteration in prompt.previous_iterations])


def _serialize_plan_with_evidence(prompt: AgentPrompt) -> str:
    if not prompt.plan_with_evidence:
        return ""
    return _serialize_json([step.model_dump() for step in prompt.plan_with_evidence])


def _append_section(parts: list[str], heading: str, content: str) -> None:
    if content:
        parts.extend([heading, content])


def _append_user_profile(parts: list[str], prompt: AgentPrompt) -> None:
    _append_section(parts, "User Profile (JSON):", _serialize_user_profile(prompt))


def _append_conversation_context(
    parts: list[str],
    prompt: AgentPrompt,
    heading: str = "Conversation Context (JSON):",
) -> None:
    if prompt.conversation_context is None:
        return
    _append_section(parts, heading, build_conversation_context_json(prompt.conversation_context))


def _append_latest_user_prompt(parts: list[str], prompt: AgentPrompt) -> None:
    _append_section(parts, "Latest User Prompt:", prompt.task)


def _join_parts(parts: list[str]) -> str:
    return "\n\n".join(part for part in parts if part)


def _build_parts(
    prompt: AgentPrompt,
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
    parts = [prompt.instruction.rstrip()]

    if include_user_profile:
        _append_user_profile(parts, prompt)
    if include_conversation_context:
        _append_conversation_context(parts, prompt, heading=conversation_context_heading)
    if include_available_tool_categories:
        _append_section(parts, "Available categories:", prompt.available_tool_categories)
    if include_available_tools:
        _append_section(parts, "Allowed Tools:", prompt.available_tools)
    if include_rules_section:
        _append_section(parts, "Rules:", prompt.rules)
    if include_rules_raw and prompt.rules:
        parts.append(prompt.rules)
    if include_previous_iterations:
        _append_section(parts, "Previous Iterations (JSON):", _serialize_previous_iterations(prompt))
    if include_plan_with_evidence:
        _append_section(parts, "Plan with Evidence (JSON):", _serialize_plan_with_evidence(prompt))
    if trailing_note:
        parts.append(trailing_note)
    if schema_as_response_label and prompt.schema:
        parts.append(f"Response Schema: {prompt.schema}")
    if include_schema_raw and prompt.schema:
        parts.append(prompt.schema)
    if include_latest_user_prompt:
        _append_latest_user_prompt(parts, prompt)

    return parts


def render_agent_prompt(prompt: AgentPrompt) -> str:
    if prompt.prompt_kind == REQUEST_ANALYSIS_PROMPT_KIND:
        return _join_parts(
            _build_parts(
                prompt,
                include_user_profile=True,
                include_conversation_context=True,
                conversation_context_heading="Conversation context (JSON):",
                include_available_tool_categories=True,
                include_latest_user_prompt=True,
                schema_as_response_label=True,
            )
        )

    if prompt.prompt_kind == PLANNER_PROMPT_KIND:
        return _join_parts(
            _build_parts(
                prompt,
                include_user_profile=True,
                include_conversation_context=True,
                include_available_tools=True,
                include_rules_raw=True,
                include_previous_iterations=True,
                include_latest_user_prompt=True,
                include_schema_raw=True,
            )
        )

    if prompt.prompt_kind == SYNTHESIS_PROMPT_KIND:
        return _join_parts(
            _build_parts(
                prompt,
                include_user_profile=True,
                include_rules_section=True,
                include_conversation_context=True,
                include_plan_with_evidence=True,
                trailing_note="Now solve the question or task according to provided evidence above.",
                include_latest_user_prompt=True,
                include_schema_raw=True,
            )
        )

    raise ValueError(f"Unsupported prompt_kind: {prompt.prompt_kind}")
