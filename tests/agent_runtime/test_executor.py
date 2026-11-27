from __future__ import annotations

import threading
from dataclasses import replace
from unittest.mock import patch
from uuid import uuid4

import pytest


from llm.conversation_model_config import MAIN_AGENT_MODEL_SCOPE
from agent_runtime.models.agent_profile import AgentExecutionStrategy
from capabilities.registry.constants import (
    TOOL_CATEGORY_FILE_MANAGEMENT,
    TOOL_CATEGORY_MATH,
)
from agent_runtime.models.agent_result import AnswerStatus, ExecutionStatus
from retrieval.evidence import ToolResult
from agent_runtime.execution.tool_executor import _agent_completion_result, _substitute_refs, run_executor
from capabilities.registry.global_registry import GLOBAL_TOOL_REGISTRY, build_agent_tool
from capabilities.discovery.models import AvailableTool, CapabilityDiscoveryResult
from langchain_core.tools import StructuredTool
from agents.document_writing.profile import DOCUMENT_WRITING_PROFILE
from agent_runtime.strategies.single_pass.graph import DirectReturnStrategy
from agent_runtime.context import (
    get_current_conversation_id,
    get_current_roundtrip_id,
    get_current_user_id,
)


def test_substitute_refs_uses_typed_tool_results() -> None:
    referenced = ToolResult(result={"value": "b"})

    assert _substitute_refs({"value": "#E2"}, {"E2": referenced}) == {
        "value": {"value": "b"}
    }
    assert _substitute_refs({"value": "#E2"}, {}) == {"value": "#E2"}


def test_subagent_execution_appends_tool_exchange_to_local_message_history(
    monkeypatch,
    agent_state_factory,
    plan_factory,
    plan_step_factory,
) -> None:
    plan = plan_factory(steps=[
        plan_step_factory(id="E1", plan="Run tool", tool="tool_a", args={}),
    ])
    state = agent_state_factory(plan=plan)
    monkeypatch.setattr(
        "agent_runtime.execution.tool_executor.call_tool",
        lambda **kwargs: ToolResult(result={"ok": True}),
    )

    run_executor(state)

    assert len(state.reasoner_message_history) == 2
    assert state.reasoner_message_history[0].tool_calls[0]["name"] == "tool_a"
    assert state.reasoner_message_history[1].name == "tool_a"


def test_run_executor_parallelizes_typed_plan_steps(
    agent_state_factory,
    plan_factory,
    plan_step_factory,
) -> None:
    for name in ("tool_a", "tool_b"):
        GLOBAL_TOOL_REGISTRY.register(StructuredTool.from_function(
            lambda: None, name=name, description="Concurrent test read", metadata={"can_run_concurrently": True},
        ))
    plan = plan_factory(steps=[
        plan_step_factory(id="E1", plan="Run tool A", tool="tool_a", args={"value": "a"}),
        plan_step_factory(id="E2", plan="Run tool B", tool="tool_b", args={"value": "b"}),
    ])
    state = agent_state_factory(plan=plan)
    started = threading.Event()
    started_steps: list[str] = []
    captured = []
    lock = threading.Lock()

    def fake_call_tool(name: str, tool_input=None, allowed_tool_names=None) -> ToolResult:
        with lock:
            started_steps.append(name)
            if len(started_steps) == 2:
                started.set()
        assert started.wait(timeout=1)
        return ToolResult(result={"tool": name, **(tool_input or {})})

    def capture_record(*args, **kwargs) -> None:
        captured.append(kwargs["execution_result"])

    with patch(
        "agent_runtime.execution.tool_executor.call_tool",
        side_effect=fake_call_tool,
    ), patch(
        "agent_runtime.execution.tool_executor._record_step_result",
        side_effect=capture_record,
    ):
        run_executor(state)

    assert sorted(started_steps) == ["tool_a", "tool_b"]
    assert [result.step.db_id for result in captured] == [step.db_id for step in plan.steps]
    assert [result.output.result for result in captured] == [
        {"tool": "tool_a", "value": "a"},
        {"tool": "tool_b", "value": "b"},
    ]


def test_run_executor_propagates_runtime_context_to_worker_threads(
    agent_state_factory,
    recording_tool_repository,
    plan_factory,
    plan_step_factory,
    execution_context_factory,
) -> None:
    GLOBAL_TOOL_REGISTRY.register(StructuredTool.from_function(
        lambda: None, name="tool_a", description="Concurrent read", metadata={"can_run_concurrently": True},
    ))
    caller_thread = threading.current_thread()
    plan = plan_factory(steps=[
        plan_step_factory(id="E1", plan="Run tool", tool="tool_a", args={}),
    ])
    state = agent_state_factory(
        plan=plan,
        execution_context=execution_context_factory(
            user_id="user-789", conversation_id="conversation-123", roundtrip_id=uuid4(),
        ),
    )
    seen_context: dict[str, str | None] = {}

    def fake_call_tool(name: str, tool_input=None, allowed_tool_names=None) -> ToolResult:
        assert threading.current_thread() is not caller_thread
        seen_context.update(
            conversation_id=get_current_conversation_id(),
            roundtrip_id=get_current_roundtrip_id(),
            user_id=get_current_user_id(),
        )
        return ToolResult(result={"ok": True})

    with patch(
        "agent_runtime.execution.tool_executor.call_tool",
        side_effect=fake_call_tool,
    ), patch("agent_runtime.execution.tool_executor._record_step_result"):
        run_executor(state)

    assert seen_context == {
        "conversation_id": "conversation-123",
        "roundtrip_id": str(state.execution_context.roundtrip_id),
        "user_id": "user-789",
    }


def test_run_executor_returns_standardized_subagent_result_without_merging_state(
    agent_profile_factory,
    monkeypatch,
    agent_state_factory,
    plan_factory,
    plan_step_factory,
) -> None:
    parent = agent_state_factory(plan=plan_factory(steps=[
        plan_step_factory(id="E1", plan="Create the document.", tool="document_writing", args={"task": "Create the document."}),
    ]))
    delegated = agent_state_factory(
        profile=agent_profile_factory(name="document_writing", scope=MAIN_AGENT_MODEL_SCOPE),
        execution_context=parent.execution_context,
    )
    parent.available_agent_states = {"document_writing": delegated}

    completed = agent_state_factory(
        profile=delegated.agent_profile,
        execution_context=parent.execution_context,
    )

    monkeypatch.setattr("agents.registry.agent_registry.get", lambda profile: lambda state: completed)

    run_executor(parent)

    assert parent.available_agent_states["document_writing"] is delegated
    assert delegated.inputs.task == ""
    assert len(parent.direct_tool_results) == 1
    assert parent.direct_tool_results[0].tool_name == "document_writing"


def test_run_executor_exposes_only_final_subagent_evaluation(
    agent_profile_factory,
    monkeypatch,
    agent_state_factory,
    plan_factory,
    plan_step_factory,
) -> None:
    parent = agent_state_factory(plan=plan_factory(steps=[
        plan_step_factory(id="E1", plan="Research the data.", tool="research", args={"task": "Research the data."}),
    ]))
    delegated = agent_state_factory(
        profile=agent_profile_factory(name="research", scope=MAIN_AGENT_MODEL_SCOPE),
        execution_context=parent.execution_context,
    )
    parent.available_agent_states = {"research": delegated}

    completed = agent_state_factory(
        profile=delegated.agent_profile,
        execution_context=parent.execution_context,
    )
    completed.node_states.evaluator.evaluation_status = "RETRYABLE"
    completed.node_states.evaluator.missing_information = ["The annual wage series."]

    monkeypatch.setattr("agents.registry.agent_registry.get", lambda profile: lambda state: completed)

    run_executor(parent)

    outcome = parent.direct_tool_results[0].state
    assert outcome is not None
    assert outcome.status == "needs_input"
    assert outcome.reason == "The annual wage series."
    assert outcome.missing_inputs == ["The annual wage series."]
    assert parent.direct_tool_results[0].result["status"] == "pending"


def test_executor_runs_explicitly_concurrent_tools_and_agents_together(
    monkeypatch,
    agent_state_factory,
    plan_factory,
    plan_step_factory,
) -> None:
    plan = plan_factory(steps=[
        plan_step_factory(id="E1", plan="Run a function", tool="batch_function", args={}),
        plan_step_factory(id="E2", plan="Run an agent", tool="document_writing", args={"task": "Create a document."}),
    ])
    parent = agent_state_factory(plan=plan)
    child = agent_state_factory(profile=DOCUMENT_WRITING_PROFILE)
    parent.available_agent_states = {"document_writing": child}
    barrier = threading.Barrier(2)

    def function():
        barrier.wait(timeout=2)
        return ToolResult(result={"value": "function output"})

    def agent(state):
        barrier.wait(timeout=2)
        state.direct_tool_results.append(ToolResult(result={"done": True}))
        return state

    registered = StructuredTool.from_function(
        function, name="batch_function", description="Test function", metadata={"can_run_concurrently": True},
    )
    monkeypatch.setitem(GLOBAL_TOOL_REGISTRY._tools, (None, registered.name), registered)
    monkeypatch.setitem(GLOBAL_TOOL_REGISTRY._tools, (None, "document_writing"), build_agent_tool(
        replace(DOCUMENT_WRITING_PROFILE, can_run_concurrently=True),
    ))
    monkeypatch.setattr("agents.registry.agent_registry.get", lambda profile: agent)

    run_executor(parent)

    assert [result.tool_name for result in parent.direct_tool_results] == ["batch_function", "document_writing"]
    assert [result.plan_step_id for result in parent.direct_tool_results] == [step.db_id for step in plan.steps]
    assert all(result.iteration == 1 for result in parent.direct_tool_results)
    assert not any(result.result.get("error") for result in parent.direct_tool_results)
    assert len(parent.reasoner_message_history) == 4


def test_executor_preserves_failures_and_duplicate_results(
    monkeypatch,
    agent_state_factory,
    plan_factory,
    plan_step_factory,
) -> None:
    plan = plan_factory(steps=[
        plan_step_factory(id=f"E{i}", plan="Run a tool", tool="broken", args={})
        for i in range(1, 4)
    ])
    state = agent_state_factory(plan=plan)
    state.available_agent_states = {}
    calls = []

    def broken(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("Provider unavailable")

    monkeypatch.setattr("agent_runtime.execution.tool_executor.call_tool", broken)
    run_executor(state)

    assert len(calls) == 2
    assert "Provider unavailable" in state.direct_tool_results[0].result["error"]
    assert state.direct_tool_results[2].result["status"] == "duplicate"
    assert [result.plan_step_id for result in state.direct_tool_results] == [step.db_id for step in plan.steps]
    assert "Provider unavailable" in state.reasoner_message_history[1].content


def test_discovery_cannot_grant_permissions_at_execution(
    agent_profile_factory,
    monkeypatch,
    agent_state_factory,
    plan_factory,
    plan_step_factory,
):
    state = agent_state_factory(
        profile=agent_profile_factory(
            name="restricted",
            scope="main_agent",
            allowed_categories={TOOL_CATEGORY_MATH, TOOL_CATEGORY_FILE_MANAGEMENT},
            excluded_categories={TOOL_CATEGORY_FILE_MANAGEMENT},
        ),
        plan=plan_factory(steps=[
            plan_step_factory(id="E1", plan="Calculate", tool="math_calculate", args={}),
            plan_step_factory(id="E2", plan="Delete", tool="file_delete", args={}),
        ]),
    )
    state.inputs.discovered_capabilities = CapabilityDiscoveryResult(tools=[
        AvailableTool(name="math_calculate", category="math"),
        AvailableTool(name="file_delete", category="math"),
    ])
    called = []

    def call_tool(**kwargs):
        called.append(kwargs["name"])
        assert kwargs["allowed_tool_names"] == state.agent_profile.tool_names
        return ToolResult(result={"ok": True})

    monkeypatch.setattr("agent_runtime.execution.tool_executor.call_tool", call_tool)
    run_executor(state)

    assert called == ["math_calculate"]
    assert state.direct_tool_results[0].result == {"ok": True}
    assert state.direct_tool_results[1].result["status"] == "rejected"
    assert "not allowed" in state.direct_tool_results[1].result["error"]


def test_executor_persists_agent_result_with_original_step_and_evidence(
    monkeypatch,
    agent_state_factory,
    recording_tool_repository,
    plan_factory,
    plan_step_factory,
    evidence_factory,
) -> None:
    plan = plan_factory(steps=[
        plan_step_factory(id="E1", plan="Write a document", tool="document_writing", args={"task": "Write a document"}),
    ])
    parent = agent_state_factory(plan=plan)
    parent.execution_context.roundtrip_id = uuid4()
    parent.agent_profile.name = "request_orchestrator"
    child = agent_state_factory(profile=DOCUMENT_WRITING_PROFILE)
    parent.available_agent_states = {"document_writing": child}
    evidence = evidence_factory(id=str(uuid4()), type="file", source="test", summary="Document saved")

    def run_child(state):
        state.direct_tool_results.append(ToolResult(result={"saved": True}, evidence=[evidence]))
        return state

    monkeypatch.setattr("agents.registry.agent_registry.get", lambda profile: run_child)
    run_executor(parent)

    call_id, payload = next(iter(recording_tool_repository.records.items()))
    step = payload["step"]
    assert step is plan.steps[0]
    assert payload["output_payload"].plan_step_id == step.db_id
    assert payload["evidence"] == [evidence]
    assert "evidence" not in payload["output_payload"].result
    assert payload["output_payload"].result["output_evidence_ids"] == [evidence.id]
    assert payload["is_subagent"] is False
    assert parent.result.tool_call_ids == [call_id]
    assert "Document saved" in parent.reasoner_message_history[1].content


def test_executor_persists_failure_status_and_direct_return_does_not_report_success(
    monkeypatch,
    agent_state_factory,
    recording_tool_repository,
    plan_factory,
    plan_step_factory,
) -> None:
    plan = plan_factory(steps=[
        plan_step_factory(id="E1", plan="Run tool", tool="broken", args={}),
    ])
    state = agent_state_factory(plan=plan)
    state.execution_context.roundtrip_id = uuid4()

    def broken(**kwargs):
        raise RuntimeError("Provider unavailable")

    monkeypatch.setattr("agent_runtime.execution.tool_executor.call_tool", broken)
    run_executor(state)
    DirectReturnStrategy._route_after_execute(state)

    payload = next(iter(recording_tool_repository.records.values()))
    assert payload["status"] == "error"
    assert "Provider unavailable" in payload["error_message"]
    assert state.result.execution_status.value == "failed"


@pytest.mark.parametrize("answer_status, execution_status, expected", [
    (AnswerStatus.SUCCESS, ExecutionStatus.SUCCESS, "completed the delegated task"),
    (AnswerStatus.PARTIAL_SUCCESS, ExecutionStatus.SUCCESS, "partially completed the delegated task"),
    (AnswerStatus.SUCCESS, ExecutionStatus.PARTIAL, "partially completed the delegated task"),
    (AnswerStatus.FAILED, ExecutionStatus.NO_NEW_WORK, "stopped because no new work was available"),
    (AnswerStatus.FAILED, ExecutionStatus.FAILED, "failed to complete the delegated task"),
    (AnswerStatus.PENDING, ExecutionStatus.PENDING, "has not completed the delegated task"),
])
def test_delegated_summary_reflects_status(
    agent_profile_factory,
    agent_state_factory,
    answer_status,
    execution_status,
    expected,
    evidence_factory,
):
    child = agent_state_factory(profile=agent_profile_factory(
        name="child", scope="main_agent", execution_strategy=AgentExecutionStrategy.REACT,
    ))
    child.result = child.result.copy(answer_status=answer_status, execution_status=execution_status)
    evidence = evidence_factory(id="ev-1", type="test", source="test", summary="Supporting finding")
    child.direct_tool_results = [ToolResult(evidence=[evidence, evidence])]

    result = _agent_completion_result(agent_name="child", state=child, task="Do the task")

    assert result.result["summary"] == f"child {expected}. Supporting finding"
    assert result.result["status"] == (
        execution_status.value if execution_status not in {ExecutionStatus.PENDING, ExecutionStatus.SUCCESS}
        else answer_status.value
    )
    assert result.evidence == [evidence]
