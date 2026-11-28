from langchain_core.tools import StructuredTool

from agent_runtime.compaction import should_compact_tool
from agent_runtime.context import bind_agent_context
from capabilities.registry.global_registry import GLOBAL_TOOL_REGISTRY


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
