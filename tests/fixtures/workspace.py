import pytest

from agent_runtime.working_memory import initialize_working_memory_file, working_memory_path
from agent_runtime.models.agent_profile import AgentExecutionStrategy
from capabilities.registry.tools import TOOL_CATEGORIES, WORKING_MEMORY_TOOLS
from common.config import UTF8_ENCODING


@pytest.fixture
def runtime_workspace(tmp_path, monkeypatch):
    """Route runtime file operations to one isolated workspace."""
    monkeypatch.setattr("assets.files.storage.conversation_workspace_path", lambda *args, **kwargs: tmp_path)
    monkeypatch.setattr("capabilities.adapters.workspace.files.conversation_workspace_path", lambda *args, **kwargs: tmp_path)
    monkeypatch.setattr("capabilities.adapters.workspace.files._workspace_root", lambda: tmp_path)
    return tmp_path


@pytest.fixture
def working_memory_state_factory(runtime_workspace, agent_state_factory, agent_profile_factory, execution_context_factory):
    def create(*, profile=None, task="", content=None, file_name=None, execution_context=None, history=(), **kwargs):
        state = agent_state_factory(
            profile=agent_profile_factory(
                name="memory_agent", execution_strategy=AgentExecutionStrategy.REACT,
                excluded_categories=set(TOOL_CATEGORIES),
                static_tools=list(WORKING_MEMORY_TOOLS),
                working_memory_template="# Working Memory\n\n## Objective\n{{ objective }}\n\n## Findings",
            ) if profile is None else profile,
            task=task,
            execution_context=(
                execution_context_factory(conversation_id="conversation-test")
                if execution_context is None else execution_context
            ),
            **kwargs,
        )
        if file_name is not None:
            state.working_memory_file_name = file_name
            path = working_memory_path(state)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(state.render_working_memory(), encoding=UTF8_ENCODING)
        initialize_working_memory_file(state)
        state.reasoner_message_history = [message.model_copy(deep=True) for message in history]
        if content is not None:
            path = working_memory_path(state)
            path.write_text(content, encoding=UTF8_ENCODING)
        return state

    return create
