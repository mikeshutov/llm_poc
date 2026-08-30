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
from request_orchestrator.agent_runner.models.agent_profile import AgentProfile
from request_orchestrator.models.agent_execution_context import AgentExecutionContext
from request_orchestrator.models.agent_state import AgentState
from request_orchestrator.models.evidence import ToolResult
from request_orchestrator.models.plan import Plan
from request_orchestrator.shared.executor.executor import _substitute_refs, run_executor
from request_orchestrator.shared.runtime_context import (
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
            extra_tools=[ToolSpec(step.tool) for step in plan.steps],
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
        "request_orchestrator.shared.executor.executor.call_tool",
        side_effect=fake_call_tool,
    ), patch(
        "request_orchestrator.shared.executor.executor._record_step_result",
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
        "request_orchestrator.shared.executor.executor.call_tool",
        side_effect=fake_call_tool,
    ), patch("request_orchestrator.shared.executor.executor._record_step_result"):
        run_executor(state)

    assert seen_context == {
        "conversation_id": "conversation-123",
        "roundtrip_id": str(state.execution_context.roundtrip_id),
        "user_id": "user-789",
    }
