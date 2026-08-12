from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

if 'yfinance' not in sys.modules:
    sys.modules['yfinance'] = ModuleType('yfinance')

if 'pycountry' not in sys.modules:
    pycountry_module = ModuleType('pycountry')
    pycountry_module.countries = SimpleNamespace(lookup=lambda value: SimpleNamespace(alpha_2=str(value).upper()))
    sys.modules['pycountry'] = pycountry_module

from request_orchestrator.agents.main_agent.router.router import router as main_router
from request_orchestrator.agents.profile_management.router.router import router as profile_router
from request_orchestrator.constants import EVALUATE_EDGE, PLAN_EDGE
from request_orchestrator.models.agent_state import AgentState, IterationState
from request_orchestrator.models.plan import Plan


def _state_with_completed_iteration(*, needs_replan: bool) -> AgentState:
    state = AgentState.new(
        task="Run tools",
        max_turns=5,
        llm=object(),
    )
    state.iteration_trace = [
        IterationState(
            plan=Plan.model_validate(
                {
                    "steps": [
                        {"id": "E1", "plan": "Run tool", "tool": "tool_a", "args": {}},
                    ]
                }
            ),
            results={"E1": {"ok": True}},
            needs_replan=needs_replan,
        )
    ]
    return state


def test_main_router_loops_back_to_planner_when_iteration_needs_replan() -> None:
    assert main_router(_state_with_completed_iteration(needs_replan=True)) == PLAN_EDGE


def test_main_router_evaluates_when_iteration_does_not_need_replan() -> None:
    assert main_router(_state_with_completed_iteration(needs_replan=False)) == EVALUATE_EDGE


def test_profile_router_loops_back_to_planner_when_iteration_needs_replan() -> None:
    assert profile_router(_state_with_completed_iteration(needs_replan=True)) == PLAN_EDGE


def test_profile_router_evaluates_when_iteration_does_not_need_replan() -> None:
    assert profile_router(_state_with_completed_iteration(needs_replan=False)) == EVALUATE_EDGE
