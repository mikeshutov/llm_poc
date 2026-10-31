from __future__ import annotations

from request_orchestrator.agent_runner.models.agent_profile import (
    AgentExecutionStrategy,
    AgentProfile,
)
from request_orchestrator.agents.registry import AgentRegistry
from request_orchestrator.models.agent_state import AgentState
from request_orchestrator.models.plan import Plan, PlanStep
from request_orchestrator.strategies.react.graph import ReactStratagy
from request_orchestrator.strategies.react.decision import DecisionResult
from request_orchestrator.strategies.react.decision import DecisionStatus


def _state(max_turns: int = 3) -> AgentState:
    return AgentState.new(
        agent_profile=AgentProfile(
            name="react_agent",
            scope="main_agent",
            execution_strategy=AgentExecutionStrategy.REACT,
            max_turns=max_turns,
        ),
        task="Find the answer",
        llm=object(),
    )


def test_react_strategy_is_registered_and_uses_planner_stage() -> None:
    assert AgentExecutionStrategy.REACT in AgentRegistry()._strategies
    assert AgentExecutionStrategy.REACT.required_model_stages() == ("planner",)


def test_react_runs_initial_action_then_planner_decision(monkeypatch) -> None:
    strategy = ReactStratagy()
    state = _state()
    calls: list[str] = []
    action = Plan(steps=[PlanStep(id="E1", plan="lookup", tool="lookup", args={})])

    def fake_decision(state, *, stage, callsite, prompt_builder):
        calls.append(stage)
        return DecisionResult(plan=action if len(calls) == 1 else Plan(steps=[]), status="CONTINUE" if len(calls) == 1 else "DONE")

    monkeypatch.setattr("request_orchestrator.strategies.react.graph.run_decision", fake_decision)
    monkeypatch.setattr("request_orchestrator.strategies.react.graph.run_executor", lambda current: current)

    result = strategy.run(state, thread_id="test")

    assert calls == ["planner", "planner"]
    assert result.node_states.planner.plan is not None
    assert result.node_states.planner.plan_count == 2


def test_react_stops_at_max_turns(monkeypatch) -> None:
    strategy = ReactStratagy()
    state = _state(max_turns=1)
    calls: list[str] = []
    action = Plan(steps=[PlanStep(id="E1", plan="lookup", tool="lookup", args={})])

    def fake_decision(state, *, stage, callsite, prompt_builder):
        calls.append(stage)
        return DecisionResult(plan=action, status="CONTINUE")

    monkeypatch.setattr("request_orchestrator.strategies.react.graph.run_decision", fake_decision)
    monkeypatch.setattr("request_orchestrator.strategies.react.graph.run_executor", lambda current: current)

    strategy.run(state, thread_id="test")

    assert calls == ["planner"]


def test_react_records_explicit_blocked_decision(monkeypatch) -> None:
    strategy = ReactStratagy()
    state = _state()

    monkeypatch.setattr(
        "request_orchestrator.strategies.react.graph.run_decision",
        lambda *args, **kwargs: DecisionResult(
            plan=Plan(steps=[]),
            status=DecisionStatus.BLOCKED,
        ),
    )

    result = strategy.run(state, thread_id="test")

    assert result.node_states.planner.decision_status == DecisionStatus.BLOCKED
    assert result.result.answer_status.value == "failed"
