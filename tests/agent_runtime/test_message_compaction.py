from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent_runtime.execution.message_history import append_execution_messages
from agent_runtime.execution.tool_executor import run_executor
from agent_runtime.context import bind_agent_context
from agent_runtime.working_memory import initialize_working_memory_file
from capabilities.adapters.workspace.working_memory import update_memory_state as memory_tool
from capabilities.registry.global_registry import call_tool
from retrieval.evidence import ToolResult


@pytest.mark.parametrize("successful", [True, False])
def test_memory_tool_itself_compacts_after_saving(working_memory_state_factory, monkeypatch, successful):
    state = working_memory_state_factory(content="Original")
    saved_contents = []
    monkeypatch.setattr(state, "condense_reasoner_history", lambda results: saved_contents.append(state.read_working_memory()))

    with bind_agent_context(state):
        if successful:
            memory_tool.invoke({"content": "Saved findings"})
        else:
            with pytest.raises(ValueError, match="not found"):
                memory_tool.invoke({"find": "Missing text", "replace": "Saved findings"})

    assert saved_contents == (["Saved findings"] if successful else [])
    assert state.read_working_memory() == ("Saved findings" if successful else "Original")


@pytest.fixture
def update_memory_state(plan_step_factory, plan_factory, runtime_workspace, monkeypatch, agent_state_factory,
                        recording_tool_repository, tool_exchange_factory):
    plan = plan_factory(steps=[
        plan_step_factory(id="search", plan="Search", tool="web_search", args={}),
        plan_step_factory(id="discover", plan="Discover", tool="discover_capabilities", args={}),
        plan_step_factory(
            id="memory",
            plan="Save viewed results",
            tool="update_memory_state",
            args={"append": "Findings"},
        ),
    ])
    state = agent_state_factory(plan=plan)
    state.agent_profile.working_memory_template = "# Working memory"
    state.execution_context.conversation_id = "conversation-test"
    state.execution_context.roundtrip_id = uuid4()
    initialize_working_memory_file(state)
    old_step = plan_step_factory(
        id="old-search",
        tool="web_search",
        plan="Previous search",
        args={"query_text": "old"},
    )
    old_id = recording_tool_repository.append_tool_call(
        state.execution_context.roundtrip_id, None, old_step, input_payload=old_step.args,
        output_payload=ToolResult(tool_name="web_search", plan_step_id=old_step.db_id, result={"old": True}),
    )
    state.result = state.result.with_recorded_tool_call(old_id)
    state.reasoner_message_history = list(tool_exchange_factory(
        tool=old_step.tool, call_id=old_step.id, args=old_step.args, content="Old search results already viewed",
    ))

    def execute(name, **kwargs):
        if name == "update_memory_state":
            return call_tool(name=name, **kwargs)
        return ToolResult(result={"new": True})

    monkeypatch.setattr("agent_runtime.execution.tool_executor.call_tool", execute)
    return state, old_id


@pytest.mark.parametrize("update_args, compacted", [
    ({"append": "Findings"}, True),
    ({"find": "missing text", "replace": "updated"}, False),
])
def test_memory_write_compacts_only_viewed_exchanges(
    conversation_message_factory,
    plan_step_factory,
    update_memory_state,
    recording_tool_repository,
    update_args,
    compacted,
):
    state, old_id = update_memory_state
    state.node_states.planner.plan.steps[-1].args = update_args
    unrelated_id = recording_tool_repository.append_tool_call(
        state.execution_context.roundtrip_id, None,
        plan_step_factory(id="other-agent", tool="web_search", plan="Unrelated search", args={}),
        input_payload={"other_agent": True}, output_payload=ToolResult(result={}),
    )
    prior_message = conversation_message_factory(content="Previous roundtrip")
    state.execution_context.conversation_context.messages = [prior_message]

    run_executor(state)

    assert recording_tool_repository.compacted_ids == ({old_id} if compacted else set())
    assert unrelated_id not in recording_tool_repository.compacted_ids
    assert not set(state.result.tool_call_ids[1:]) & recording_tool_repository.compacted_ids
    names = [message.name for message in state.reasoner_message_history if isinstance(message, ToolMessage)]
    assert names == (["web_search", "discover_capabilities", "update_memory_state"] if compacted else [
        "web_search", "web_search", "discover_capabilities", "update_memory_state",
    ])
    assert len(state.gather_tool_results()) == 4  # Audit/evidence reads retain compacted results.
    assert state.execution_context.conversation_context.messages == [prior_message]


def test_appending_results_does_not_itself_compact_history(
    plan_factory,
    plan_step_factory,
    agent_state_factory,
):
    plan = plan_factory(steps=[
        plan_step_factory(id="search", tool="web_search", plan="Search", args={}),
        plan_step_factory(id="memory", tool="update_memory_state", plan="Save", args={"append": "Findings"}),
    ])
    state = agent_state_factory(plan=plan)
    state.direct_tool_results = [
        ToolResult(tool_name=step.tool, plan_step_id=step.db_id, result={"ok": True}) for step in plan.steps
    ]

    append_execution_messages(state)

    assert [message.name for message in state.reasoner_message_history if isinstance(message, ToolMessage)] == [
        "web_search", "update_memory_state",
    ]


def test_read_phase_sees_saved_memory_and_compacted_prior_history(
    update_memory_state,
    recording_tool_repository,
    monkeypatch,
):
    state, old_id = update_memory_state
    reads = []

    def execute(name, **kwargs):
        if name == "update_memory_state":
            return call_tool(name=name, **kwargs)
        assert "Findings" in state.read_working_memory()
        assert recording_tool_repository.compacted_ids == {old_id}
        assert not state.reasoner_message_history
        assert state.result.tool_call_ids == [old_id]  # No current-batch records exist yet.
        reads.append(name)
        return ToolResult(result={"new": True})

    # Discovery is itself serial, so leave only the read and memory update.
    state.node_states.planner.plan.steps = [
        step for step in state.node_states.planner.plan.steps if step.tool != "discover_capabilities"
    ]
    monkeypatch.setattr("agent_runtime.execution.tool_executor.call_tool", execute)

    run_executor(state)

    assert reads == ["web_search"]
    assert [message.name for message in state.reasoner_message_history if isinstance(message, ToolMessage)] == [
        "web_search", "update_memory_state",
    ]


def test_multiple_memory_writes_never_compact_current_batch_results(
    plan_step_factory,
    update_memory_state,
    recording_tool_repository,
):
    state, old_id = update_memory_state
    state.node_states.planner.plan.steps.append(plan_step_factory(
        id="second-memory", tool="update_memory_state", plan="Save again", args={"append": "More findings"},
    ))

    run_executor(state)

    assert recording_tool_repository.compacted_ids == {old_id}
    assert [message.name for message in state.reasoner_message_history if isinstance(message, ToolMessage)] == [
        "web_search", "discover_capabilities", "update_memory_state", "update_memory_state",
    ]


def test_agent_state_condenses_mixed_calls_but_retains_excluded_tools(agent_state_factory, tool_exchange_factory):
    state = agent_state_factory()
    user = HumanMessage(content="Find an answer")
    assistant = AIMessage(content="", tool_calls=[
        {"id": "search", "name": "web_search", "args": {}},
        {"id": "discover", "name": "discover_capabilities", "args": {}},
    ])
    discovery_result = ToolMessage(content="Tools", tool_call_id="discover", name="discover_capabilities")
    memory_call, memory_result = tool_exchange_factory(tool="update_memory_state", call_id="memory", content="Saved")
    state.reasoner_message_history = [
        user, assistant, ToolMessage(content="Results", tool_call_id="search", name="web_search"),
        discovery_result, memory_call, memory_result,
    ]
    history = state.reasoner_message_history

    state.condense_reasoner_history([])

    assert state.reasoner_message_history is history
    assert history[0] is user
    assert [call["id"] for call in history[1].tool_calls] == ["discover"]
    assert [call["id"] for call in assistant.tool_calls] == ["search", "discover"]
    assert history[2:] == [discovery_result, memory_call, memory_result]
