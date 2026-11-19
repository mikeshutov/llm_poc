from __future__ import annotations

from uuid import uuid4

from assets.files.storage import conversation_workspace_path
from agent_runtime.models.agent_state import AgentState


def initialize_working_memory_file(state: AgentState) -> None:
    """Create the delegate's private runtime memory file before it runs."""
    conversation_id = state.execution_context.conversation_id
    if not conversation_id:
        return
    user_id = state.execution_context.user_profile.user_id
    file_name = f".delegate-working-memory-{uuid4().hex}.md"
    directory = conversation_workspace_path(conversation_id, user_id=user_id)
    directory.mkdir(parents=True, exist_ok=True)
    content = state.agent_profile.working_memory_template.replace(
        "{{ objective }}",
        state.inputs.task or state.inputs.request_task,
    ).replace("{memory_path}", file_name)
    (directory / file_name).write_text(content, encoding="utf-8")
    state.working_memory_file_name = file_name
