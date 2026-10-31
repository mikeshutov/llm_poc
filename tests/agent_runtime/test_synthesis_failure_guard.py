from __future__ import annotations

from agent_runtime.models.agent_result import AnswerStatus, ExecutionStatus
from agent_runtime.models.agent_state import AgentState
from agent_runtime.models.main_state import MainState
from agent_runtime.models.plan import Plan, PlanStep
from agent_runtime.synthesis.synthesis import run_synthesis
from agents.main.request import TOP_LEVEL_PROFILE


def test_synthesis_does_not_invoke_llm_after_failed_execution_without_evidence(monkeypatch) -> None:
    state = MainState.new(
        task="Find a product",
        llm=object(),
        agent_profiles=[TOP_LEVEL_PROFILE],
    )
    agent_state = AgentState.new(
        agent_profile=TOP_LEVEL_PROFILE,
        task="Find a product",
        llm=object(),
    )
    agent_state.node_states.planner.plan = Plan(
        steps=[PlanStep(id="E1", plan="Search", tool="generic_web_search", args={})]
    )
    agent_state.result = agent_state.result.copy(
        answer_status=AnswerStatus.FAILED,
        execution_status=ExecutionStatus.FAILED,
    )
    state.main_agent_state = agent_state

    def fail_if_called(*args, **kwargs):
        raise AssertionError("synthesis LLM must not be called")

    monkeypatch.setattr("llm.chat_models.build_chat_model", fail_if_called)

    result = run_synthesis(state)

    assert result.result.answer == [
        "I couldn't find any information for that request. Try rephrasing."
    ]
    assert result.result.result_blocks == []


def test_synthesis_does_not_invoke_llm_for_empty_plan_without_current_evidence(monkeypatch) -> None:
    state = MainState.new(
        task="Give me recipes",
        llm=object(),
        agent_profiles=[TOP_LEVEL_PROFILE],
    )
    agent_state = AgentState.new(
        agent_profile=TOP_LEVEL_PROFILE,
        task="Give me recipes",
        llm=object(),
    )
    agent_state.node_states.planner.plan = Plan(steps=[])
    state.main_agent_state = agent_state

    def fail_if_called(*args, **kwargs):
        raise AssertionError("synthesis LLM must not be called")

    monkeypatch.setattr("llm.chat_models.build_chat_model", fail_if_called)

    result = run_synthesis(state)

    assert result.result.answer == [
        "I couldn't find any information for that request. Try rephrasing."
    ]
    assert result.result.result_blocks == []
