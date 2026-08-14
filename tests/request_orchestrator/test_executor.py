from __future__ import annotations

import sys
import threading
import time
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

if 'yfinance' not in sys.modules:
    sys.modules['yfinance'] = ModuleType('yfinance')

if 'pycountry' not in sys.modules:
    pycountry_module = ModuleType('pycountry')
    pycountry_module.countries = SimpleNamespace(lookup=lambda value: SimpleNamespace(alpha_2=str(value).upper()))
    sys.modules['pycountry'] = pycountry_module

from conversation.models.conversation_model_config import MAIN_AGENT_MODEL_SCOPE
from request_orchestrator.models.agent_profile import AgentProfile
from request_orchestrator.models.agent_state import AgentState, IterationState
from request_orchestrator.models.evidence import ToolResult
from request_orchestrator.models.plan import Plan
from request_orchestrator.shared.executor.executor import run_executor
from request_orchestrator.shared.runtime_context import bind_runtime_context, get_current_conversation_id, get_current_roundtrip_id, get_current_user_id


class RecordingRepo:
    def __init__(self) -> None:
        self.conversation_events: list[dict] = []

    def create_conversation_event(self, **kwargs):
        self.conversation_events.append(kwargs)
        return kwargs


def test_run_executor_parallelizes_pending_steps() -> None:
    profile = AgentProfile(
        name="test_agent",
        scope=MAIN_AGENT_MODEL_SCOPE,
        extra_tools=[
            SimpleNamespace(name="tool_a"),
            SimpleNamespace(name="tool_b"),
        ],
    )
    state = AgentState.new(
        task="Run tools",
        max_turns=5,
        llm=object(),
        agent_profile=profile,
        conversation_id=str(uuid4()),
    )
    state.iteration_trace = [
        IterationState(
            plan=Plan.model_validate(
                {
                    "steps": [
                        {"id": "E1", "plan": "Run tool A", "tool": "tool_a", "args": {"value": "a"}},
                        {"id": "E2", "plan": "Run tool B", "tool": "tool_b", "args": {"value": "b"}},
                    ]
                }
            ),
            results={},
        )
    ]

    lock = threading.Lock()
    started_steps: list[str] = []
    released = threading.Event()

    def fake_call_tool(name: str, tool_input=None, allowed_tool_names=None):
        step_name = str(name)
        with lock:
            started_steps.append(step_name)
            if len(started_steps) == 2:
                released.set()
        assert released.wait(timeout=1.0)
        time.sleep(0.02)
        return ToolResult(result={"tool": step_name, **(tool_input or {})})

    with patch(
        "request_orchestrator.shared.executor.executor.call_tool",
        side_effect=fake_call_tool,
    ), patch(
        'common.logging.conversation_event_logger.get_conversation_repo',
        return_value=RecordingRepo(),
        ) as repo_getter:
                with bind_runtime_context(
                    conversation_id=state.conversation_id or str(uuid4()),
                    conversation_model_config=state.conversation_model_config,
                    roundtrip_id=None,
                    user_id=None,
                ):
                    run_executor(state)

    assert state.iteration_trace[-1].results["P1E1"].result == {"tool": "tool_a", "value": "a"}
    assert state.iteration_trace[-1].results["P1E2"].result == {"tool": "tool_b", "value": "b"}
    assert started_steps[:2] == ["tool_a", "tool_b"]
    repo = repo_getter.return_value
    assert len(repo.conversation_events) == 2
    assert all(isinstance(event.get("payload", {}).get("data", {}).get("latency_ms"), int) for event in repo.conversation_events)


def test_run_executor_leaves_unresolved_refs_unchanged_with_parallel_execution() -> None:
    profile = AgentProfile(
        name="test_agent",
        scope=MAIN_AGENT_MODEL_SCOPE,
        extra_tools=[
            SimpleNamespace(name="tool_a"),
            SimpleNamespace(name="tool_b"),
        ],
    )
    state = AgentState.new(
        task="Run tools",
        max_turns=5,
        llm=object(),
        agent_profile=profile,
    )
    state.iteration_trace = [
        IterationState(
            plan=Plan.model_validate(
                {
                    "steps": [
                        {"id": "E1", "plan": "Run tool A", "tool": "tool_a", "args": {"value": "#E2"}},
                        {"id": "E2", "plan": "Run tool B", "tool": "tool_b", "args": {"value": "b"}},
                    ]
                }
            ),
            results={},
        )
    ]

    seen_inputs: dict[str, dict[str, object]] = {}

    def fake_call_tool(name: str, tool_input=None, allowed_tool_names=None):
        seen_inputs[str(name)] = dict(tool_input or {})
        return ToolResult(result={"tool": str(name), **(tool_input or {})})

    with patch(
        "request_orchestrator.shared.executor.executor.call_tool",
        side_effect=fake_call_tool,
    ):
        run_executor(state)

    assert seen_inputs["tool_a"] == {"value": "#E2"}
    assert seen_inputs["tool_b"] == {"value": "b"}


def test_run_executor_unwraps_prior_step_result_for_resolved_refs() -> None:
    profile = AgentProfile(
        name="test_agent",
        scope=MAIN_AGENT_MODEL_SCOPE,
        extra_tools=[
            SimpleNamespace(name="tool_a"),
            SimpleNamespace(name="tool_b"),
        ],
    )
    state = AgentState.new(
        task="Run tools",
        max_turns=5,
        llm=object(),
        agent_profile=profile,
    )
    state.iteration_trace = [
        IterationState(
            plan=Plan.model_validate(
                {
                    "steps": [
                        {"id": "E1", "plan": "Run tool A", "tool": "tool_a", "args": {"value": "a"}},
                    ]
                }
            ),
            results={"P1E2": ToolResult(result={"value": "b"})},
        )
    ]

    seen_inputs: dict[str, dict[str, object]] = {}
    state.iteration_trace[-1].plan = Plan.model_validate(
        {
            "steps": [
                {"id": "E1", "plan": "Run tool A", "tool": "tool_a", "args": {"value": "#E2"}},
            ]
        }
    )

    def fake_call_tool(name: str, tool_input=None, allowed_tool_names=None):
        seen_inputs[str(name)] = dict(tool_input or {})
        return ToolResult(result={"tool": str(name), **(tool_input or {})})

    with patch(
        "request_orchestrator.shared.executor.executor.call_tool",
        side_effect=fake_call_tool,
    ):
        run_executor(state)

    assert seen_inputs["tool_a"] == {"value": {"value": "b"}}


def test_run_executor_propagates_runtime_context_to_worker_threads() -> None:
    profile = AgentProfile(
        name="test_agent",
        scope=MAIN_AGENT_MODEL_SCOPE,
        extra_tools=[SimpleNamespace(name="tool_a")],
    )
    state = AgentState.new(
        task="Run tools",
        max_turns=5,
        llm=object(),
        agent_profile=profile,
        conversation_id="conversation-123",
    )
    state.roundtrip_id = "roundtrip-456"  # type: ignore[assignment]
    state.user_profile.user_id = "user-789"
    state.iteration_trace = [
        IterationState(
            plan=Plan.model_validate(
                {
                    "steps": [
                        {"id": "E1", "plan": "Run tool A", "tool": "tool_a", "args": {"value": "a"}},
                    ]
                }
            ),
            results={},
        )
    ]

    seen_context: dict[str, str | None] = {}

    def fake_call_tool(name: str, tool_input=None, allowed_tool_names=None):
        seen_context["conversation_id"] = get_current_conversation_id()
        seen_context["roundtrip_id"] = get_current_roundtrip_id()
        seen_context["user_id"] = get_current_user_id()
        return ToolResult(result={"tool": str(name), **(tool_input or {})})

    with patch(
        "request_orchestrator.shared.executor.executor.call_tool",
        side_effect=fake_call_tool,
    ):
        run_executor(state)

    assert seen_context == {
        "conversation_id": "conversation-123",
        "roundtrip_id": "roundtrip-456",
        "user_id": "user-789",
    }
