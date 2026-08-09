from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

if 'yfinance' not in sys.modules:
    sys.modules['yfinance'] = ModuleType('yfinance')

if 'pycountry' not in sys.modules:
    pycountry_module = ModuleType('pycountry')
    pycountry_module.countries = SimpleNamespace(lookup=lambda value: SimpleNamespace(alpha_2=str(value).upper()))
    sys.modules['pycountry'] = pycountry_module

from langgraph.graph import END

from request_orchestrator.agents.profile_management.router.router import router
from request_orchestrator.constants import EVALUATE_EDGE, EXECUTE_TOOLS_EDGE, PLAN_EDGE
from request_orchestrator.models.agent_state import AgentState, IterationState
from request_orchestrator.models.plan import Plan
from request_orchestrator.shared.evaluator import evaluator_router


def test_profile_router_routes_first_pass_to_plan() -> None:
    state = AgentState.new(task="Remember this", max_turns=5, llm=object())

    assert router(state) == PLAN_EDGE


def test_profile_router_routes_empty_plan_to_end() -> None:
    state = AgentState.new(task="Remember this", max_turns=5, llm=object())
    state.iteration_trace = [
        IterationState(
            plan=Plan.model_validate({"steps": []}),
            results={},
        )
    ]

    assert router(state) == END


def test_profile_router_routes_pending_steps_to_execute() -> None:
    state = AgentState.new(task="Remember this", max_turns=5, llm=object())
    state.iteration_trace = [
        IterationState(
            plan=Plan.model_validate({
                "steps": [
                    {
                        "id": "E1",
                        "plan": "Load current profile state.",
                        "tool": "get_user_attributes",
                        "args": {"limit": 10, "is_active": True}}
                ]}),
            results={},
        )
    ]

    assert router(state) == EXECUTE_TOOLS_EDGE


def test_profile_router_routes_completed_results_to_evaluator() -> None:
    state = AgentState.new(task="Remember this", max_turns=5, llm=object())
    state.iteration_trace = [
        IterationState(
            plan=Plan.model_validate({
                "steps": [
                    {
                        "id": "E1",
                        "plan": "Load current profile state.",
                        "tool": "get_user_attributes",
                        "args": {"limit": 10, "is_active": True}}
                ]}),
            results={"E1": {"items": []}},
        )
    ]

    assert router(state) == EVALUATE_EDGE


def test_profile_evaluator_router_can_end_when_satisfied() -> None:
    state = AgentState.new(task="Remember this", max_turns=5, llm=object())
    state.goal_reached = True

    assert evaluator_router(state) == 'synthesize'
