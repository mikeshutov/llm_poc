from __future__ import annotations

from types import SimpleNamespace


from agents.main.request import TOP_LEVEL_PROFILE
from agent_runtime.routing.main_router import router
from agent_runtime.strategies.plan_execute_evaluate.result_validator import (
    execution_result_router,
    run_execution_result_validator,
)
from agent_runtime.strategies.plan_execute_evaluate.validator import validator
from agent_runtime.constants import EVALUATE_EDGE, EXECUTE_TOOLS_EDGE, PLAN_EDGE, SYNTHESIZE_EDGE
from agent_runtime.models.agent_result import AnswerStatus, ExecutionStatus
from agent_runtime.models.evaluation_result import (
    EVALUATION_STATUS_PARTIAL_BUT_SUFFICIENT,
    EVALUATION_STATUS_RETRYABLE,
    EVALUATION_STATUS_SATISFIED,
    EVALUATION_STATUS_TERMINAL,
)
from retrieval.evidence import EvidenceView, ToolResult
from agent_runtime.models.plan import Plan
from agent_runtime.evaluation import evaluator_router


def test_validator_routes_empty_plan_to_evaluator(set_plan_results, agent_state_factory) -> None:
    state = agent_state_factory(task="Find something", llm=object(), profile=TOP_LEVEL_PROFILE)
    set_plan_results(state, plan=Plan.model_validate({"steps": []}), results={}, plan_count=1)

    assert validator(state) == EVALUATE_EDGE


def test_validator_routes_missing_plan_to_evaluator(agent_state_factory) -> None:
    state = agent_state_factory(task="Find something", llm=object(), profile=TOP_LEVEL_PROFILE)
    state.node_states.planner.plan = None

    assert validator(state) == EVALUATE_EDGE


def test_validator_routes_action_plan_to_execute(set_plan_results, agent_state_factory) -> None:
    state = agent_state_factory(task="Find something", llm=object(), profile=TOP_LEVEL_PROFILE)
    set_plan_results(
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


def test_validator_executes_new_action_plan_even_if_evaluator_is_terminal(
    set_plan_results,
    agent_state_factory,
) -> None:
    state = agent_state_factory(task="Find something", llm=object(), profile=TOP_LEVEL_PROFILE)
    set_plan_results(
        state,
        plan=Plan.model_validate({
            "steps": [{
                "id": "E1",
                "plan": "Look it up",
                "tool": "generic_web_search",
                "args": {"query_text": "okonomiyaki kit"},
            }]
        }),
        results={},
        plan_count=1,
    )
    state.node_states.evaluator.evaluation_status = EVALUATION_STATUS_TERMINAL

    assert validator(state) == EXECUTE_TOOLS_EDGE


def test_router_routes_executed_results_to_evaluator(set_plan_results, agent_state_factory) -> None:
    state = agent_state_factory(task="Find something", llm=object(), profile=TOP_LEVEL_PROFILE)
    set_plan_results(
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


def test_router_routes_missing_results_back_to_plan(set_plan_results, agent_state_factory) -> None:
    state = agent_state_factory(task="Find something", llm=object(), profile=TOP_LEVEL_PROFILE)
    set_plan_results(
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


def test_execution_result_router_replans_once_after_an_empty_execution(
    set_plan_results,
    agent_state_factory,
) -> None:
    state = agent_state_factory(task="Find something", llm=object(), profile=TOP_LEVEL_PROFILE)
    set_plan_results(
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

    assert state.result.answer_status is AnswerStatus.FAILED
    assert state.result.execution_status is ExecutionStatus.SUCCESS
    assert state.node_states.planner.no_result_attempts == 1
    assert execution_result_router(state) == PLAN_EDGE


def test_execution_result_router_evaluates_after_second_empty_execution(
    set_plan_results,
    agent_state_factory,
) -> None:
    state = agent_state_factory(task="Find something", llm=object(), profile=TOP_LEVEL_PROFILE)
    set_plan_results(
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

    assert state.result.answer_status is AnswerStatus.FAILED
    assert state.node_states.planner.no_result_attempts == 2
    assert execution_result_router(state) == EVALUATE_EDGE


def test_execution_result_router_replans_before_evaluation_when_results_exist(
    set_plan_results,
    agent_state_factory,
) -> None:
    state = agent_state_factory(task="Find something", llm=object(), profile=TOP_LEVEL_PROFILE)
    set_plan_results(
        state,
        plan=Plan.model_validate({
            "steps": [{
                "id": "E1",
                "plan": "Look it up",
                "tool": "generic_web_search",
                "args": {"query_text": "okonomiyaki kit"},
            }]
        }),
        results={"E1": {"results": ["evidence"]}},
        plan_count=1,
    )
    state.gather_tool_calls = lambda: [
        SimpleNamespace(plan_step_id=state.node_states.planner.plan.steps[0].db_id, status="completed")
    ]
    state.gather_tool_results = lambda: [
        ToolResult(
            plan_step_id=state.node_states.planner.plan.steps[0].db_id,
            tool_name="generic_web_search",
            result={"results": ["evidence"]},
            evidence=[EvidenceView(
        data={
            "title": "Evidence",
        })],
        )
    ]

    run_execution_result_validator(state)

    assert execution_result_router(state) == PLAN_EDGE


def test_evaluator_router_requires_state_save_when_status_is_satisfied(agent_state_factory) -> None:
    state = agent_state_factory(task="Find something", llm=object(), profile=TOP_LEVEL_PROFILE)
    state.node_states.evaluator.evaluation_status = EVALUATION_STATUS_SATISFIED

    assert evaluator_router(state) == PLAN_EDGE


def test_evaluator_router_requires_state_save_when_status_is_partial_but_sufficient(
    agent_state_factory,
) -> None:
    state = agent_state_factory(task="Find something", llm=object(), profile=TOP_LEVEL_PROFILE)
    state.node_states.evaluator.evaluation_status = EVALUATION_STATUS_PARTIAL_BUT_SUFFICIENT

    assert evaluator_router(state) == PLAN_EDGE


def test_evaluator_router_returns_synthesis_when_status_is_terminal(agent_state_factory) -> None:
    state = agent_state_factory(task="Find something", llm=object(), profile=TOP_LEVEL_PROFILE)
    state.node_states.evaluator.evaluation_status = EVALUATION_STATUS_TERMINAL

    assert evaluator_router(state) == SYNTHESIZE_EDGE


def test_evaluator_router_returns_plan_when_status_is_retryable(agent_state_factory) -> None:
    state = agent_state_factory(task="Find something", llm=object(), profile=TOP_LEVEL_PROFILE)
    state.node_states.evaluator.evaluation_status = EVALUATION_STATUS_RETRYABLE

    assert evaluator_router(state) == PLAN_EDGE
    assert state.node_states.evaluator.evaluation_status == EVALUATION_STATUS_RETRYABLE


