from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

if 'yfinance' not in sys.modules:
    sys.modules['yfinance'] = ModuleType('yfinance')

if 'pycountry' not in sys.modules:
    pycountry_module = ModuleType('pycountry')
    pycountry_module.countries = SimpleNamespace(lookup=lambda value: SimpleNamespace(alpha_2=str(value).upper()))
    sys.modules['pycountry'] = pycountry_module

from agents.main.request import TOP_LEVEL_PROFILE
from agent_runtime.routing.main_router import router as main_router
from agent_runtime.constants import EVALUATE_EDGE
from agent_runtime.models.agent_state import AgentState
from retrieval.evidence import ToolResult
from agent_runtime.models.plan import Plan


def _state_with_completed_plan(*, needs_replan: bool, profile=TOP_LEVEL_PROFILE) -> AgentState:
    state = AgentState.new(
        task="Run tools",
        llm=object(),
        agent_profile=profile,
    )
    plan = Plan.model_validate(
        {
            "steps": [
                {"id": "E1", "plan": "Run tool", "tool": "tool_a", "args": {}},
            ]
        }
    )
    state.node_states.planner.plan = plan
    state.node_states.planner.plan_count = 1
    state.node_states.planner.needs_replan = needs_replan
    state.gather_tool_results = lambda: [
        ToolResult(
            plan_step_id=plan.steps[0].db_id,
            tool_name="tool_a",
            result={"ok": True},
        )
    ]
    return state


def test_main_router_ignores_needs_replan_and_evaluates() -> None:
    assert main_router(_state_with_completed_plan(needs_replan=True)) == EVALUATE_EDGE


def test_main_router_evaluates_when_plan_does_not_need_replan() -> None:
    assert main_router(_state_with_completed_plan(needs_replan=False)) == EVALUATE_EDGE
