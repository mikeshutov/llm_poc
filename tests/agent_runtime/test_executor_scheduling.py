import threading

import pytest
from langchain_core.tools import StructuredTool

from agent_runtime.context import bind_agent_context, current_agent_state
from agent_runtime.execution.tool_executor import run_executor
from capabilities.registry.global_registry import GLOBAL_TOOL_REGISTRY
from capabilities.registry.models import Tool
from retrieval.evidence import ToolResult


@pytest.mark.parametrize("write_fails", [False, True])
def test_serial_writes_finish_before_concurrent_reads_in_plan_order(
    plan_factory,
    plan_step_factory,
    agent_state_factory,
    write_fails,
):
    events = []
    reads = threading.Barrier(2)
    writes_finished = threading.Event()

    def write_first():
        events.append("write_first")
        assert "read_a" not in events and "read_b" not in events
        if write_fails:
            raise ValueError("Write failed")
        return ToolResult(result={"saved": True})

    def write_second():
        assert events == ["write_first"]
        events.append("write_second")
        writes_finished.set()
        return ToolResult(result={"saved": True})

    def read(name):
        assert writes_finished.is_set()
        active_state = current_agent_state.get()
        assert active_state is not None
        assert active_state.execution_context.conversation_id == "scheduling-test"
        events.append(name)
        reads.wait(timeout=2)
        return ToolResult(result={"read": name})

    functions = {
        "read_a": lambda: read("read_a"),
        "write_first": write_first,
        "read_b": lambda: read("read_b"),
        "write_second": write_second,
    }
    for name, fn in functions.items():
        raw = StructuredTool.from_function(fn, name=name, description="Scheduling test")
        GLOBAL_TOOL_REGISTRY.register(Tool(raw, can_run_concurrently=name.startswith("read_")))
    plan = plan_factory(
        steps=[plan_step_factory(id=name, tool=name, plan=name, args={}) for name in functions],
    )
    state = agent_state_factory(plan=plan)
    state.execution_context.conversation_id = "scheduling-test"

    run_executor(state)

    assert events[:2] == ["write_first", "write_second"]
    assert set(events[2:]) == {"read_a", "read_b"}
    assert [result.tool_name for result in state.direct_tool_results] == list(functions)
    assert [result.status for result in state.direct_tool_results] == [
        "completed", "error" if write_fails else "completed", "completed", "completed",
    ]


def test_tools_without_concurrency_declaration_are_serial(
    plan_factory,
    plan_step_factory,
    agent_state_factory,
    monkeypatch,
):
    events = []
    for name in ("first", "second"):
        GLOBAL_TOOL_REGISTRY.register(StructuredTool.from_function(
            lambda name=name: events.append(name), name=name, description="Undeclared tool",
        ))
    plan = plan_factory(
        steps=[plan_step_factory(id=name, tool=name, plan=name, args={}) for name in ("first", "second")],
    )
    state = agent_state_factory(plan=plan)

    def unexpected_pool(**kwargs):
        raise AssertionError("Serial-only batches must not allocate a thread pool")

    monkeypatch.setattr("agent_runtime.execution.tool_executor.ThreadPoolExecutor", unexpected_pool)

    run_executor(state)

    assert events == ["first", "second"]


def test_registry_concurrency_policy_respects_user_scope(agent_state_factory):
    tool = StructuredTool.from_function(
        lambda: None, name="user_read", description="User read", metadata={"can_run_concurrently": True},
    )
    GLOBAL_TOOL_REGISTRY.register(tool, user_id="alice")
    state = agent_state_factory()
    state.execution_context.user_profile.user_id = "alice"
    with bind_agent_context(state):
        assert GLOBAL_TOOL_REGISTRY.can_run_concurrently(tool.name)
    state.execution_context.user_profile.user_id = "bob"
    with bind_agent_context(state):
        assert not GLOBAL_TOOL_REGISTRY.can_run_concurrently(tool.name)


@pytest.mark.parametrize("name, concurrent", [
    ("web_search", True), ("workspace_file_read", True), ("file_list", True),
    ("workspace_file_complete", False), ("workspace_file_write", False), ("file_edit", False),
    ("code_execute", False), ("discover_capabilities", False), ("document_writing", False),
    ("unknown_tool", False),
])
def test_builtin_concurrency_declarations(name, concurrent):
    assert GLOBAL_TOOL_REGISTRY.can_run_concurrently(name) is concurrent
