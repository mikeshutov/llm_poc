from __future__ import annotations

from agent_runtime.models.agent_result import AnswerStatus, ExecutionStatus
from agent_runtime.synthesis.synthesis import run_synthesis
from agents.main.request import TOP_LEVEL_PROFILE


def test_synthesis_does_not_invoke_llm_after_failed_execution_without_evidence(
    plan_factory,
    plan_step_factory,
    main_state_factory,
    agent_state_factory,
    monkeypatch,
) -> None:
    state = main_state_factory(
        task="Find a product",
        llm=object(),
        agent_profiles=[TOP_LEVEL_PROFILE],
    )
    agent_state = agent_state_factory(
        profile=TOP_LEVEL_PROFILE,
        task="Find a product",
        llm=object(),
    )
    agent_state.node_states.planner.plan = plan_factory(
        steps=[plan_step_factory(id="E1", plan="Search", tool="generic_web_search", args={})]
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


def test_synthesis_does_not_invoke_llm_for_empty_plan_without_current_evidence(
    plan_factory,
    main_state_factory,
    agent_state_factory,
    monkeypatch,
) -> None:
    state = main_state_factory(
        task="Give me recipes",
        llm=object(),
        agent_profiles=[TOP_LEVEL_PROFILE],
    )
    agent_state = agent_state_factory(
        profile=TOP_LEVEL_PROFILE,
        task="Give me recipes",
        llm=object(),
    )
    agent_state.node_states.planner.plan = plan_factory(steps=[])
    state.main_agent_state = agent_state

    def fail_if_called(*args, **kwargs):
        raise AssertionError("synthesis LLM must not be called")

    monkeypatch.setattr("llm.chat_models.build_chat_model", fail_if_called)

    result = run_synthesis(state)

    assert result.result.answer == [
        "I couldn't find any information for that request. Try rephrasing."
    ]
    assert result.result.result_blocks == []
