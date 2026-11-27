from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from common.config import UTF8_ENCODING
from assets.files.storage import (
    WORKING_MEMORY_DIRECTORY,
    conversation_working_memory_path,
)
from agent_runtime.compaction import is_successful_memory_update
from retrieval.evidence import ToolResult

if TYPE_CHECKING:
    from agent_runtime.models.agent_state import AgentState

MAX_WORKING_MEMORY_BYTES = 32_000
_TRUNCATION_MARKER = "\n[Working memory truncated; consolidate it with update_memory_state.]"


def bound_working_memory(content: str) -> str:
    encoded = content.encode(UTF8_ENCODING)
    if len(encoded) <= MAX_WORKING_MEMORY_BYTES:
        return content
    budget = MAX_WORKING_MEMORY_BYTES - len(_TRUNCATION_MARKER.encode(UTF8_ENCODING))
    return encoded[:budget].decode(UTF8_ENCODING, errors="ignore") + _TRUNCATION_MARKER


def validate_working_memory(content: str) -> None:
    if len(content.encode(UTF8_ENCODING)) > MAX_WORKING_MEMORY_BYTES:
        raise ValueError(f"Working memory exceeds {MAX_WORKING_MEMORY_BYTES} bytes; consolidate it before saving.")


def working_memory_path(state: AgentState) -> Path | None:
    if not state.execution_context.conversation_id or not state.working_memory_file_name:
        return None
    return conversation_working_memory_path(
        state.execution_context.conversation_id,
        state.working_memory_file_name,
        user_id=state.execution_context.user_profile.user_id,
    )


def render_working_memory_template(state: AgentState) -> str:
    return state.agent_profile.working_memory_template.replace(
        "{{ objective }}", state.inputs.task or state.inputs.request_task,
    )


def task_completion_requested(result: ToolResult | None) -> bool:
    """Return whether a successful memory update requested agent completion."""
    if result is None or not is_successful_memory_update(result):
        return False
    return isinstance(result.result, dict) and result.result.get("task_completed") is True


def initialize_working_memory_file(state: AgentState) -> None:
    """Create a profile-driven private working-memory file before an agent runs."""
    template = state.agent_profile.working_memory_template
    conversation_id = state.execution_context.conversation_id
    if not template or not conversation_id:
        return

    existing_path = working_memory_path(state)
    if existing_path is not None and existing_path.is_file():
        return

    content = render_working_memory_template(state)
    validate_working_memory(content)

    profile_slug = re.sub(r"[^a-z0-9]+", "-", state.agent_profile.name.lower()).strip("-") or "agent"
    file_name = f".agent-working-memory-{profile_slug}-{uuid4().hex}.md"
    memory_file_name = f"{WORKING_MEMORY_DIRECTORY}/{file_name}"
    path = conversation_working_memory_path(
        conversation_id, memory_file_name, user_id=state.execution_context.user_profile.user_id,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding=UTF8_ENCODING)
    state.working_memory_file_name = memory_file_name
