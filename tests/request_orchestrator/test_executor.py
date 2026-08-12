from __future__ import annotations

import sys
import threading
import time
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

if 'yfinance' not in sys.modules:
    sys.modules['yfinance'] = ModuleType('yfinance')

if 'pycountry' not in sys.modules:
    pycountry_module = ModuleType('pycountry')
    pycountry_module.countries = SimpleNamespace(lookup=lambda value: SimpleNamespace(alpha_2=str(value).upper()))
    sys.modules['pycountry'] = pycountry_module

from request_orchestrator.agents.models.agent_profile import AgentProfile
from request_orchestrator.models.agent_state import AgentState, IterationState
from request_orchestrator.models.plan import Plan
from request_orchestrator.shared.executor.executor import run_executor


def test_run_executor_parallelizes_pending_steps() -> None:
    profile = AgentProfile(
        name="test_agent",
        extra_tools=[
            SimpleNamespace(name="tool_a"),
            SimpleNamespace(name="tool_b"),
        ],
        persist_tool_calls=False,
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
        return {"tool": step_name, **(tool_input or {})}

    with patch(
        "request_orchestrator.shared.executor.executor.call_tool",
        side_effect=fake_call_tool,
    ):
        run_executor(state)

    assert state.iteration_trace[-1].results["P1E1"] == {"tool": "tool_a", "value": "a"}
    assert state.iteration_trace[-1].results["P1E2"] == {"tool": "tool_b", "value": "b"}
    assert started_steps[:2] == ["tool_a", "tool_b"]
    assert len(state.agent_log.entries) == 2
    assert all(isinstance(entry.data.get("latency_ms"), int) for entry in state.agent_log.entries)


def test_run_executor_leaves_unresolved_refs_unchanged_with_parallel_execution() -> None:
    profile = AgentProfile(
        name="test_agent",
        extra_tools=[
            SimpleNamespace(name="tool_a"),
            SimpleNamespace(name="tool_b"),
        ],
        persist_tool_calls=False,
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
        return {"tool": str(name), **(tool_input or {})}

    with patch(
        "request_orchestrator.shared.executor.executor.call_tool",
        side_effect=fake_call_tool,
    ):
        run_executor(state)

    assert seen_inputs["tool_a"] == {"value": "#E2"}
    assert seen_inputs["tool_b"] == {"value": "b"}
