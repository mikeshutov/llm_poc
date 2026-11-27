import pytest
from langchain_core.tools import StructuredTool

from agent_runtime.compaction import is_successful_memory_update, should_compact_tool
from agent_runtime.context import bind_agent_context
from agent_runtime.working_memory import task_completion_requested
from capabilities.registry.global_registry import GLOBAL_TOOL_REGISTRY
from retrieval.evidence import ToolOutcome, ToolOutcomeStatus, ToolResult


@pytest.mark.parametrize("payload, outcome, status", [
    ({"task_completed": True}, None, "completed"),
    ({"status": "failed", "task_completed": True}, None, "error"),
    ({"error": "write failed", "task_completed": True}, None, "error"),
    ({"error_message": "write failed", "task_completed": True}, None, "error"),
    ({"status": "rejected", "task_completed": True}, None, "rejected"),
    ({"status": "duplicate", "task_completed": True}, None, "rejected"),
    ({"task_completed": True}, ToolOutcome(status=ToolOutcomeStatus.FAILED), "error"),
])
def test_memory_compaction_and_completion_use_common_result_status(payload, outcome, status):
    result = ToolResult(tool_name="update_memory_state", result=payload, state=outcome)

    assert result.status == status
    assert is_successful_memory_update(result) is (status == "completed")
    assert task_completion_requested(result) is (status == "completed")


def test_registered_runtime_tool_compaction_respects_user_scope(agent_state_factory):
    def invoke() -> dict:
        """Return a result that must remain in history."""
        return {}

    tool = StructuredTool.from_function(invoke, name="persistent_runtime_tool", metadata={"compactable": False})
    GLOBAL_TOOL_REGISTRY.register(tool, user_id="alice")
    state = agent_state_factory()
    state.execution_context.user_profile.user_id = "alice"

    with bind_agent_context(state):
        assert should_compact_tool(tool.name) is False
        assert should_compact_tool("discover_capabilities") is False
        assert should_compact_tool("update_memory_state") is False
        assert should_compact_tool("unregistered_tool") is False
