import importlib

import pytest

from agent_runtime.context import bind_agent_context
from agent_runtime.models.agent_execution_context import AgentExecutionContext
from agent_runtime.working_memory import MAX_WORKING_MEMORY_BYTES, initialize_working_memory_file
from agents.delegate.profile import DELEGATE_PROFILE
from capabilities.adapters.workspace.working_memory import update_memory_state
from capabilities.registry.global_registry import GLOBAL_TOOL_REGISTRY
from pydantic import ValidationError


def test_memory_tool_is_registered_as_serial_only():
    registered = GLOBAL_TOOL_REGISTRY.get("update_memory_state")
    assert registered.version == 7
    assert registered.can_run_concurrently is False


@pytest.mark.parametrize("args", [
    {}, {"content": "new", "append": "extra"}, {"append": "extra", "regex": True},
])
def test_memory_tool_requires_exactly_one_valid_update_mode(working_memory_state_factory, args):
    state = working_memory_state_factory(content="Original")
    with bind_agent_context(state), pytest.raises(ValidationError):
        update_memory_state.invoke(args)
    assert state.read_working_memory() == "Original"


@pytest.mark.parametrize("mode", ["content", "append", "find"])
@pytest.mark.parametrize("character", ["x", "é"])
def test_memory_tool_rejects_oversized_updates(working_memory_state_factory, mode, character):
    state = working_memory_state_factory(content="Original")
    large = character * MAX_WORKING_MEMORY_BYTES + "x"
    args = {mode: large} if mode != "find" else {"find": "Original", "replace": large}
    with bind_agent_context(state), pytest.raises(ValueError, match="consolidate"):
        update_memory_state.invoke(args)
    assert state.read_working_memory() == "Original"


def test_memory_read_bounds_externally_modified_files(working_memory_state_factory):
    state = working_memory_state_factory(content="x" * (MAX_WORKING_MEMORY_BYTES + 1))
    contents = state.read_working_memory()
    assert len(contents.encode("utf-8")) <= MAX_WORKING_MEMORY_BYTES
    assert "truncated" in contents


def test_working_memory_tools_are_scoped_to_runtime_file(
    runtime_workspace,
    working_memory_state_factory,
) -> None:
    module = importlib.import_module("capabilities.adapters.workspace.working_memory")
    state = working_memory_state_factory(
        file_name="memory/delegate-memory.md", content="# Working Memory\n\n## Findings\n",
    )
    memory_file = runtime_workspace / state.working_memory_file_name

    with bind_agent_context(state):
        update_result = module.update_memory_state.invoke({
            "content": "# Working Memory\n\nUpdated",
        })
        completion_result = module.update_memory_state.invoke({
            "append": "\n\n## Decisions\nKeep it concise.",
            "task_completed": True,
        })

    assert memory_file.read_text(encoding="utf-8") == "# Working Memory\n\nUpdated\n\n## Decisions\nKeep it concise."
    assert update_result.evidence[0].data["title"] == "memory/delegate-memory.md"
    assert completion_result.result["task_completed"] is True


def test_update_memory_state_supports_exact_and_regex_replacement(
    runtime_workspace,
    working_memory_state_factory,
) -> None:
    module = importlib.import_module("capabilities.adapters.workspace.working_memory")
    state = working_memory_state_factory(content="Finding: source-123")
    memory_file = runtime_workspace / state.working_memory_file_name

    with bind_agent_context(state):
        module.update_memory_state.invoke({"find": "source-123", "replace": "source-456"})
        module.update_memory_state.invoke({"find": r"source-\d+", "replace": "source-final", "regex": True})

    assert memory_file.read_text(encoding="utf-8") == "Finding: source-final"


def test_working_memory_tools_do_not_create_unbound_files(runtime_workspace, agent_state_factory) -> None:
    module = importlib.import_module("capabilities.adapters.workspace.working_memory")
    state = agent_state_factory(profile=DELEGATE_PROFILE)
    state.working_memory_file_name = "missing-memory.md"
    state.execution_context = AgentExecutionContext.new(conversation_id="conversation-1")

    with bind_agent_context(state):
        try:
            module.update_memory_state.invoke({"content": "should not be created"})
        except ValueError as exc:
            assert "runtime must create it first" in str(exc)
        else:
            raise AssertionError("update_memory_state created a file outside runtime initialization")

    assert not (runtime_workspace / "missing-memory.md").exists()


def test_oversized_template_is_rejected_before_file_creation(runtime_workspace, agent_state_factory):
    state = agent_state_factory(
        profile_overrides={"working_memory_template": "x" * (MAX_WORKING_MEMORY_BYTES + 1)},
    )
    state.execution_context.conversation_id = "conversation-test"
    with pytest.raises(ValueError, match="consolidate"):
        initialize_working_memory_file(state)
    assert state.working_memory_file_name is None
    assert not list(runtime_workspace.rglob("*.md"))


@pytest.mark.parametrize("character", ["x", "é"])
def test_memory_write_accepts_exact_byte_limit(working_memory_state_factory, character):
    state = working_memory_state_factory()
    contents = character * (MAX_WORKING_MEMORY_BYTES // len(character.encode("utf-8")))
    with bind_agent_context(state):
        update_memory_state.invoke({"content": contents})
    assert state.read_working_memory() == contents
    assert len(contents.encode("utf-8")) == MAX_WORKING_MEMORY_BYTES


def test_initializing_memory_again_preserves_saved_content(working_memory_state_factory):
    state = working_memory_state_factory(content="Existing findings")
    filename = state.working_memory_file_name
    initialize_working_memory_file(state)
    assert state.working_memory_file_name == filename
    assert state.read_working_memory() == "Existing findings"
