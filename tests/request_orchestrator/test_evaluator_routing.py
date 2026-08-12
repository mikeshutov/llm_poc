from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

if 'yfinance' not in sys.modules:
    sys.modules['yfinance'] = ModuleType('yfinance')

if 'pycountry' not in sys.modules:
    pycountry_module = ModuleType('pycountry')
    pycountry_module.countries = SimpleNamespace(lookup=lambda value: SimpleNamespace(alpha_2=str(value).upper()))
    sys.modules['pycountry'] = pycountry_module

from request_orchestrator.agents.main_agent.router.router import router
from request_orchestrator.agents.main_agent.validator.validator import validator
from request_orchestrator.constants import EVALUATE_EDGE, EXECUTE_TOOLS_EDGE, PLAN_EDGE, SYNTHESIZE_EDGE
from request_orchestrator.models.agent_state import AgentState, IterationState
from request_orchestrator.models.evaluation_result import EVALUATION_STATUS_RETRYABLE, EVALUATION_STATUS_SATISFIED, EVALUATION_STATUS_TERMINAL
from request_orchestrator.models.evaluation_result import (
    EVALUATION_STATUS_RETRYABLE,
    EVALUATION_STATUS_SATISFIED,
    EVALUATION_STATUS_TERMINAL,
)
from request_orchestrator.models.plan import Plan
from request_orchestrator.shared.evaluator import evaluator_router


def test_validator_routes_empty_plan_to_synthesis() -> None:
    state = AgentState.new(task="Find something", max_turns=5, llm=object())
    state.iteration_trace = [
        IterationState(
            plan=Plan.model_validate({"steps": []}),
            results={},
        )
    ]

    assert validator(state) == SYNTHESIZE_EDGE


def test_validator_routes_empty_plan_to_synthesis_again() -> None:
    state = AgentState.new(task="Find something", max_turns=5, llm=object())
    state.iteration_trace = [
        IterationState(
            plan=Plan.model_validate({"steps": []}),
            results={},
        )
    ]

    assert validator(state) == SYNTHESIZE_EDGE


def test_validator_routes_action_plan_to_execute() -> None:
    state = AgentState.new(task="Find something", max_turns=5, llm=object())
    state.iteration_trace = [
        IterationState(
            plan=Plan.model_validate({
                "steps": [
                    {
                        "id": "E1",
                        "plan": "Look it up",
                        "tool": "generic_web_search",
                        "args": {"query_text": "okonomiyaki kit"}}
                ]}),
            results={},
        )
    ]

    assert validator(state) == EXECUTE_TOOLS_EDGE


def test_router_routes_executed_results_to_evaluator() -> None:
    state = AgentState.new(task="Find something", max_turns=5, llm=object())
    state.iteration_trace = [
        IterationState(
            plan=Plan.model_validate({
                "steps": [
                    {
                        "id": "E1",
                        "plan": "Look it up",
                        "tool": "generic_web_search",
                        "args": {"query_text": "okonomiyaki kit"}}
                ]}),
            results={"P1E1": {"items": []}},
        )
    ]

    assert router(state) == EVALUATE_EDGE


def test_router_routes_missing_results_back_to_plan() -> None:
    state = AgentState.new(task="Find something", max_turns=5, llm=object())
    state.iteration_trace = [
        IterationState(
            plan=Plan.model_validate({
                "steps": [
                    {
                        "id": "E1",
                        "plan": "Look it up",
                        "tool": "generic_web_search",
                        "args": {"query_text": "okonomiyaki kit"}}
                ]}),
            results={},
        )
    ]

    assert router(state) == PLAN_EDGE


def test_evaluator_router_returns_synthesis_when_status_is_satisfied() -> None:
    state = AgentState.new(task="Find something", max_turns=5, llm=object())
    state.evaluation_status = EVALUATION_STATUS_SATISFIED

    assert evaluator_router(state) == SYNTHESIZE_EDGE


def test_evaluator_router_returns_synthesis_when_status_is_terminal() -> None:
    state = AgentState.new(task="Find something", max_turns=5, llm=object())
    state.evaluation_status = EVALUATION_STATUS_TERMINAL

    assert evaluator_router(state) == SYNTHESIZE_EDGE


def test_evaluator_router_returns_plan_when_status_is_retryable() -> None:
    state = AgentState.new(task="Find something", max_turns=5, llm=object())
    state.evaluation_status = EVALUATION_STATUS_RETRYABLE
    state.goal_reached = False

    assert evaluator_router(state) == PLAN_EDGE
