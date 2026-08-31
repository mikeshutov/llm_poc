from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from langgraph.graph import END

if 'yfinance' not in sys.modules:
    sys.modules['yfinance'] = ModuleType('yfinance')

if 'pycountry' not in sys.modules:
    pycountry_module = ModuleType('pycountry')
    pycountry_module.countries = SimpleNamespace(lookup=lambda value: SimpleNamespace(alpha_2=str(value).upper()))
    sys.modules['pycountry'] = pycountry_module

from request_orchestrator.agents.main_agent.profile import MAIN_AGENT_PROFILE
from request_orchestrator.agents.main_agent.router.router import router
from request_orchestrator.agent_runner.stratagies.planner_executor_evaluator.result_validator import (
    execution_result_router,
    run_execution_result_validator,
)
from request_orchestrator.agent_runner.stratagies.planner_executor_evaluator.validator import validator
from request_orchestrator.constants import EVALUATE_EDGE, EXECUTE_TOOLS_EDGE, PLAN_EDGE, SYNTHESIZE_EDGE
from request_orchestrator.models.agent_execution_context import AgentExecutionContext
from request_orchestrator.models.agent_result import ResultStatus
from request_orchestrator.models.agent_state import AgentState
from request_orchestrator.models.evaluation_result import (
    EVALUATION_STATUS_RETRYABLE,
    EVALUATION_STATUS_SATISFIED,
    EVALUATION_STATUS_TERMINAL,
)
from request_orchestrator.models.evidence import ToolResult
from request_orchestrator.models.plan import Plan, PlanStep
from request_orchestrator.shared.evaluator import evaluator_router


def _hydrate_plan_state(
    state: AgentState,
    *,
    plan: Plan,
    results: dict[str, object] | None = None,
    plan_count: int = 1,
    needs_replan: bool = False,
) -> None:
    state.node_states.planner.plan = plan.model_copy(deep=True)
    state.node_states.planner.plan_count = max(0, plan_count)
    state.node_states.planner.needs_replan = needs_replan
    step_by_local_id = {step.id: step for step in plan.steps}
    normalized_results: list[ToolResult] = []
    for local_step_id, value in (results or {}).items():
        step = step_by_local_id[local_step_id]
        if isinstance(value, ToolResult):
            normalized_results.append(
                value.model_copy(
                    update={
                        "plan_step_id": value.plan_step_id or step.db_id,
                        "tool_name": value.tool_name or step.tool,
                    }
                )
            )
            continue
        normalized_results.append(
            ToolResult(
                plan_step_id=step.db_id,
                tool_name=step.tool,
                result=value,
            )
        )
    state.gather_tool_results = lambda: normalized_results


def test_validator_routes_empty_plan_to_synthesis() -> None:
    state = AgentState.new(task="Find something", llm=object(), agent_profile=MAIN_AGENT_PROFILE)
    _hydrate_plan_state(state, plan=Plan.model_validate({"steps": []}), results={}, plan_count=1)

    assert validator(state) == SYNTHESIZE_EDGE


def test_validator_routes_empty_plan_to_synthesis_again() -> None:
    state = AgentState.new(task="Find something", llm=object(), agent_profile=MAIN_AGENT_PROFILE)
    _hydrate_plan_state(state, plan=Plan.model_validate({"steps": []}), results={}, plan_count=1)

    assert validator(state) == SYNTHESIZE_EDGE


def test_validator_routes_action_plan_to_execute() -> None:
    state = AgentState.new(task="Find something", llm=object(), agent_profile=MAIN_AGENT_PROFILE)
    _hydrate_plan_state(
        state,
        plan=Plan.model_validate({
            "steps": [
                {
                    "id": "E1",
                    "plan": "Look it up",
                    "tool": "generic_web_search",
                    "args": {"query_text": "okonomiyaki kit"}}
            ]}),
        results={},
        plan_count=1,
    )

    assert validator(state) == EXECUTE_TOOLS_EDGE


def test_router_routes_executed_results_to_evaluator() -> None:
    state = AgentState.new(task="Find something", llm=object(), agent_profile=MAIN_AGENT_PROFILE)
    _hydrate_plan_state(
        state,
        plan=Plan.model_validate({
            "steps": [
                {
                    "id": "E1",
                    "plan": "Look it up",
                    "tool": "generic_web_search",
                    "args": {"query_text": "okonomiyaki kit"}}
            ]}),
        results={"E1": {"items": []}},
        plan_count=1,
    )

    assert router(state) == EVALUATE_EDGE


def test_router_routes_missing_results_back_to_plan() -> None:
    state = AgentState.new(task="Find something", llm=object(), agent_profile=MAIN_AGENT_PROFILE)
    _hydrate_plan_state(
        state,
        plan=Plan.model_validate({
            "steps": [
                {
                    "id": "E1",
                    "plan": "Look it up",
                    "tool": "generic_web_search",
                    "args": {"query_text": "okonomiyaki kit"}}
            ]}),
        results={},
        plan_count=1,
    )

    assert router(state) == PLAN_EDGE


def test_execution_result_router_replans_once_after_an_empty_execution() -> None:
    state = AgentState.new(task="Find something", llm=object(), agent_profile=MAIN_AGENT_PROFILE)
    _hydrate_plan_state(
        state,
        plan=Plan.model_validate({
            "steps": [
                {
                    "id": "E1",
                    "plan": "Look it up",
                    "tool": "generic_web_search",
                    "args": {"query_text": "okonomiyaki kit"},
                }
            ]
        }),
        results={},
        plan_count=1,
    )
    state.gather_tool_calls = lambda: [
        SimpleNamespace(plan_step_id=state.node_states.planner.plan.steps[0].db_id, status="completed")
    ]
    state.gather_tool_results = lambda: [
        ToolResult(
            plan_step_id=state.node_states.planner.plan.steps[0].db_id,
            tool_name="generic_web_search",
            result={"results": []},
        )
    ]

    run_execution_result_validator(state)

    assert state.result.result_status is ResultStatus.FAILED
    assert state.node_states.planner.no_result_attempts == 1
    assert execution_result_router(state) == PLAN_EDGE


def test_execution_result_router_skips_evaluator_after_second_empty_execution() -> None:
    state = AgentState.new(task="Find something", llm=object(), agent_profile=MAIN_AGENT_PROFILE)
    _hydrate_plan_state(
        state,
        plan=Plan.model_validate({
            "steps": [
                {
                    "id": "E1",
                    "plan": "Look it up again",
                    "tool": "generic_web_search",
                    "args": {"query_text": "okonomiyaki kit"},
                }
            ]
        }),
        results={},
        plan_count=2,
    )
    state.node_states.planner.no_result_attempts = 1
    state.gather_tool_calls = lambda: [
        SimpleNamespace(plan_step_id=state.node_states.planner.plan.steps[0].db_id, status="completed")
    ]
    state.gather_tool_results = lambda: [
        ToolResult(
            plan_step_id=state.node_states.planner.plan.steps[0].db_id,
            tool_name="generic_web_search",
            result={"results": []},
        )
    ]

    run_execution_result_validator(state)

    assert state.result.result_status is ResultStatus.FAILED
    assert state.node_states.planner.no_result_attempts == 2
    assert execution_result_router(state) == END


def test_evaluator_router_returns_synthesis_when_status_is_satisfied() -> None:
    state = AgentState.new(task="Find something", llm=object(), agent_profile=MAIN_AGENT_PROFILE)
    state.node_states.evaluator.evaluation_status = EVALUATION_STATUS_SATISFIED

    assert evaluator_router(state) == SYNTHESIZE_EDGE


def test_evaluator_router_returns_synthesis_when_status_is_terminal() -> None:
    state = AgentState.new(task="Find something", llm=object(), agent_profile=MAIN_AGENT_PROFILE)
    state.node_states.evaluator.evaluation_status = EVALUATION_STATUS_TERMINAL

    assert evaluator_router(state) == SYNTHESIZE_EDGE


def test_evaluator_router_returns_plan_when_status_is_retryable() -> None:
    state = AgentState.new(task="Find something", llm=object(), agent_profile=MAIN_AGENT_PROFILE)
    state.node_states.evaluator.evaluation_status = EVALUATION_STATUS_RETRYABLE
    state.node_states.evaluator.goal_reached = False

    assert evaluator_router(state) == PLAN_EDGE


def test_main_agent_graph_executes_plan_after_planner(monkeypatch) -> None:
    from request_orchestrator.agents.main_agent import agent as main_agent_module
    from request_orchestrator.agent_runner.stratagies.planner_executor_evaluator import graph as strategy_module

    planner_called = False
    executor_called = False

    def fake_planner(state: AgentState) -> AgentState:
        nonlocal planner_called
        planner_called = True
        _hydrate_plan_state(
            state,
            plan=Plan(
                steps=[
                    PlanStep(
                        id="E1",
                        plan="Look it up",
                        tool="generic_web_search",
                        args={"query_text": "okonomiyaki kit"},
                    )
                ]
            ),
            results={},
            plan_count=1,
        )
        return state

    def fake_executor(state: AgentState) -> AgentState:
        nonlocal executor_called
        executor_called = True
        state.node_states.evaluator.goal_reached = True
        return state

    monkeypatch.setattr(strategy_module, "run_planner", fake_planner)
    monkeypatch.setattr(strategy_module, "run_executor", fake_executor)

    final_state = main_agent_module.run_agent(
        user_query="Find something",
        execution_context=AgentExecutionContext.new(),
        llm=object(),
    )

    assert planner_called is True
    assert executor_called is True
    assert final_state.node_states.evaluator.goal_reached is True
