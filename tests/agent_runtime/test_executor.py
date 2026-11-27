from __future__ import annotations

import threading
from dataclasses import dataclass
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

if "yfinance" not in sys.modules:
    sys.modules["yfinance"] = ModuleType("yfinance")

if "pycountry" not in sys.modules:
    pycountry_module = ModuleType("pycountry")
    pycountry_module.countries = SimpleNamespace(
        lookup=lambda value: SimpleNamespace(alpha_2=str(value).upper())
    )
    sys.modules["pycountry"] = pycountry_module

from llm.conversation_model_config import MAIN_AGENT_MODEL_SCOPE
from agent_runtime.models.agent_profile import AgentProfile
from agent_runtime.models.agent_execution_context import AgentExecutionContext
from agent_runtime.models.agent_state import AgentState
from retrieval.evidence import ToolResult
from agent_runtime.models.plan import Plan
from agent_runtime.execution.tool_executor import _substitute_refs, run_executor
from agent_runtime.execution.message_history import append_execution_messages
from capabilities.registry.global_registry import register_default_tools
from agent_runtime.context import (
    get_current_conversation_id,
    get_current_roundtrip_id,
    get_current_user_id,
)


@dataclass(frozen=True)
class ToolSpec:
    name: str


def _state_for_plan(plan: Plan) -> AgentState:
    state = AgentState.new(
        task="Run tools",
        llm=object(),
        agent_profile=AgentProfile(
            name="test_agent",
            scope=MAIN_AGENT_MODEL_SCOPE,
            static_tools=[ToolSpec(step.tool) for step in plan.steps],
        ),
    )
    state.node_states.planner.plan = plan
    state.node_states.planner.plan_count = 1
    return state


def test_substitute_refs_uses_typed_tool_results() -> None:
    referenced = ToolResult(result={"value": "b"})

    assert _substitute_refs({"value": "#E2"}, {"E2": referenced}) == {
        "value": {"value": "b"}
    }
    assert _substitute_refs({"value": "#E2"}, {}) == {"value": "#E2"}


def test_subagent_execution_appends_tool_exchange_to_local_message_history(monkeypatch) -> None:
    plan = Plan.model_validate({
        "steps": [{"id": "E1", "plan": "Run tool", "tool": "tool_a", "args": {}}]
    })
    state = _state_for_plan(plan)
    monkeypatch.setattr(
        "agent_runtime.execution.tool_executor.call_tool",
        lambda **kwargs: ToolResult(result={"ok": True}),
    )

    run_executor(state)

    assert len(state.reasoner_message_history) == 2
    assert state.reasoner_message_history[0].tool_calls[0]["name"] == "tool_a"
    assert state.reasoner_message_history[1].name == "tool_a"


def test_run_executor_parallelizes_typed_plan_steps() -> None:
    plan = Plan.model_validate(
        {
            "steps": [
                {"id": "E1", "plan": "Run tool A", "tool": "tool_a", "args": {"value": "a"}},
                {"id": "E2", "plan": "Run tool B", "tool": "tool_b", "args": {"value": "b"}},
            ]
        }
    )
    state = _state_for_plan(plan)
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

    assert started_steps == ["tool_a", "tool_b"]
    assert [result.step.db_id for result in captured] == [step.db_id for step in plan.steps]
    assert [result.output.result for result in captured] == [
        {"tool": "tool_a", "value": "a"},
        {"tool": "tool_b", "value": "b"},
    ]


def test_run_executor_propagates_runtime_context_to_worker_threads() -> None:
    plan = Plan.model_validate(
        {"steps": [{"id": "E1", "plan": "Run tool", "tool": "tool_a", "args": {}}]}
    )
    state = _state_for_plan(plan)
    state.execution_context = AgentExecutionContext.new(conversation_id="conversation-123")
    state.execution_context.roundtrip_id = uuid4()
    state.execution_context.user_profile.user_id = "user-789"
    seen_context: dict[str, str | None] = {}

    def fake_call_tool(name: str, tool_input=None, allowed_tool_names=None) -> ToolResult:
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


def test_run_executor_returns_standardized_subagent_result_without_merging_state(monkeypatch) -> None:
    register_default_tools()
    parent = _state_for_plan(Plan.model_validate({
        "steps": [{
            "id": "E1",
            "plan": "Create the document.",
            "tool": "document_writing",
            "args": {"task": "Create the document."},
        }],
    }))
    delegated = AgentState.new(
        task="",
        agent_profile=AgentProfile(name="document_writing", scope=MAIN_AGENT_MODEL_SCOPE),
        execution_context=parent.execution_context,
        llm=object(),
    )
    parent.available_agent_states = {"document_writing": delegated}

    completed = AgentState.new(
        task="",
        agent_profile=delegated.agent_profile,
        execution_context=parent.execution_context,
        llm=object(),
    )

    class Registry:
        @staticmethod
        def get(profile):
            return lambda state: completed

    monkeypatch.setattr("agents.registry.agent_registry", Registry())

    run_executor(parent)

    assert parent.available_agent_states["document_writing"] is delegated
    assert delegated.inputs.task == ""
    assert len(parent.direct_tool_results) == 1
    assert parent.direct_tool_results[0].tool_name == "document_writing"


def test_run_executor_exposes_only_final_subagent_evaluation(monkeypatch) -> None:
    register_default_tools()
    parent = _state_for_plan(Plan.model_validate({
        "steps": [{
            "id": "E1",
            "plan": "Research the data.",
            "tool": "research",
            "args": {"task": "Research the data."},
        }],
    }))
    delegated = AgentState.new(
        task="",
        agent_profile=AgentProfile(name="research", scope=MAIN_AGENT_MODEL_SCOPE),
        execution_context=parent.execution_context,
        llm=object(),
    )
    parent.available_agent_states = {"research": delegated}

    completed = AgentState.new(
        task="",
        agent_profile=delegated.agent_profile,
        execution_context=parent.execution_context,
        llm=object(),
    )
    completed.node_states.evaluator.evaluation_status = "RETRYABLE"
    completed.node_states.evaluator.missing_information = ["The annual wage series."]

    class Registry:
        @staticmethod
        def get(profile):
            return lambda state: completed

    monkeypatch.setattr("agents.registry.agent_registry", Registry())

    run_executor(parent)

    outcome = parent.direct_tool_results[0].state
    assert outcome is not None
    assert outcome.status == "needs_input"
    assert outcome.reason == "The annual wage series."
    assert outcome.missing_inputs == ["The annual wage series."]
    assert parent.direct_tool_results[0].result["status"] == "pending"


def test_executor_runs_tools_and_agents_in_one_parallel_batch(monkeypatch) -> None:
    from langchain_core.tools import StructuredTool
    from capabilities.registry.global_registry import GLOBAL_TOOL_REGISTRY
    from agents.document_writing.profile import DOCUMENT_WRITING_PROFILE

    register_default_tools()
    plan = Plan.model_validate({"steps": [
        {"id": "E1", "plan": "Run a function", "tool": "batch_function", "args": {}},
        {"id": "E2", "plan": "Run an agent", "tool": "document_writing", "args": {"task": "Create a document."}},
    ]})
    parent = _state_for_plan(plan)
    child = AgentState.new(agent_profile=DOCUMENT_WRITING_PROFILE, llm=object())
    parent.available_agent_states = {"document_writing": child}
    barrier = threading.Barrier(2)

    def function():
        barrier.wait(timeout=2)
        return ToolResult(result={"value": "function output"})

    def agent(state):
        barrier.wait(timeout=2)
        state.direct_tool_results.append(ToolResult(result={"done": True}))
        return state

    registered = StructuredTool.from_function(function, name="batch_function", description="Test function")
    monkeypatch.setitem(GLOBAL_TOOL_REGISTRY._tools, (None, registered.name), registered)
    monkeypatch.setattr("agents.registry.agent_registry.get", lambda profile: agent)

    run_executor(parent)

    assert [result.tool_name for result in parent.direct_tool_results] == ["batch_function", "document_writing"]
    assert [result.plan_step_id for result in parent.direct_tool_results] == [step.db_id for step in plan.steps]
    assert all(result.iteration == 1 for result in parent.direct_tool_results)
    assert not any(result.result.get("error") for result in parent.direct_tool_results)
    assert len(parent.reasoner_message_history) == 4


def test_executor_preserves_failures_and_duplicate_results(monkeypatch) -> None:
    plan = Plan.model_validate({"steps": [
        {"id": f"E{i}", "plan": "Run a tool", "tool": "broken", "args": {}}
        for i in range(1, 4)
    ]})
    state = _state_for_plan(plan)
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


def test_executor_persists_agent_result_with_original_step_and_evidence(monkeypatch) -> None:
    from agents.document_writing.profile import DOCUMENT_WRITING_PROFILE
    from retrieval.evidence import EvidenceView

    register_default_tools()
    plan = Plan.model_validate({"steps": [{
        "id": "E1", "plan": "Write a document", "tool": "document_writing",
        "args": {"task": "Write a document"},
    }]})
    parent = _state_for_plan(plan)
    parent.execution_context.roundtrip_id = uuid4()
    parent.agent_profile.name = "request_orchestrator"
    child = AgentState.new(agent_profile=DOCUMENT_WRITING_PROFILE, llm=object())
    parent.available_agent_states = {"document_writing": child}
    evidence = EvidenceView(id=str(uuid4()), type="file", source="test", summary="Document saved")
    recorded = []
    call_id = uuid4()

    class Repository:
        def count_request_hash(self, *args):
            return 0

        def append_tool_call(self, roundtrip_id, current_plan, step, **kwargs):
            recorded.append((step, kwargs))
            return call_id

        def get_tool_results(self, ids):
            return [recorded[0][1]["output_payload"]]

    def run_child(state):
        state.direct_tool_results.append(ToolResult(result={"saved": True}, evidence=[evidence]))
        return state

    repo = Repository()
    monkeypatch.setattr("agents.registry.agent_registry.get", lambda profile: run_child)
    monkeypatch.setattr("agent_runtime.execution.tool_executor.MessageToolRepository", lambda: repo)
    monkeypatch.setattr("agent_runtime.models.agent_state.message_repository.MessageToolRepository", lambda: repo)
    run_executor(parent)

    step, payload = recorded[0]
    assert step is plan.steps[0]
    assert payload["output_payload"].plan_step_id == step.db_id
    assert payload["evidence"] == [evidence]
    assert "evidence" not in payload["output_payload"].result
    assert payload["output_payload"].result["output_evidence_ids"] == [evidence.id]
    assert payload["is_subagent"] is False
    assert parent.result.tool_call_ids == [call_id]
    assert "Document saved" in parent.reasoner_message_history[1].content


def test_executor_persists_failure_status_and_direct_return_does_not_report_success(monkeypatch) -> None:
    from agent_runtime.strategies.single_pass.graph import DirectReturnStrategy

    plan = Plan.model_validate({"steps": [{
        "id": "E1", "plan": "Run tool", "tool": "broken", "args": {},
    }]})
    state = _state_for_plan(plan)
    state.execution_context.roundtrip_id = uuid4()
    recorded = []

    class Repository:
        def count_request_hash(self, *args):
            return 0

        def append_tool_call(self, *args, **kwargs):
            recorded.append(kwargs)
            return uuid4()

        def get_tool_results(self, ids):
            return [entry["output_payload"] for entry in recorded]

    def broken(**kwargs):
        raise RuntimeError("Provider unavailable")

    repo = Repository()
    monkeypatch.setattr("agent_runtime.execution.tool_executor.call_tool", broken)
    monkeypatch.setattr("agent_runtime.execution.tool_executor.MessageToolRepository", lambda: repo)
    monkeypatch.setattr("agent_runtime.models.agent_state.message_repository.MessageToolRepository", lambda: repo)
    run_executor(state)
    DirectReturnStrategy._route_after_execute(state)

    assert recorded[0]["status"] == "error"
    assert "Provider unavailable" in recorded[0]["error_message"]
    assert state.result.execution_status.value == "failed"
