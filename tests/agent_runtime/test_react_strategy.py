from __future__ import annotations

import pytest

from agent_runtime.models.agent_profile import AgentExecutionStrategy, AgentKind
from agents.registry import AgentRegistry
from agent_runtime.strategies.react.graph import ReactStrategy
from agent_runtime.strategies.single_pass.graph import DirectReturnStrategy
from agent_runtime.strategies.react.decision import DecisionResult
from agent_runtime.models.decision import DecisionStatus
from retrieval.evidence import ToolResult


def test_react_strategy_is_registered_and_uses_planner_stage() -> None:
    assert AgentExecutionStrategy.REACT in AgentRegistry()._strategies
    assert AgentExecutionStrategy.REACT.required_model_stages() == ("planner",)


def test_react_runs_initial_action_then_planner_decision(react_agent_state_factory, plan_factory, plan_step_factory, monkeypatch) -> None:
    strategy = ReactStrategy()
    state = react_agent_state_factory()
    calls: list[str] = []
    action = plan_factory(steps=[plan_step_factory(id="E1", plan="lookup", tool="lookup", args={})])

    def fake_decision(state, *, stage, callsite, prompt_builder, message_history=None):
        calls.append(stage)
        return DecisionResult(
            plan=action if len(calls) == 1 else plan_factory(steps=[]),
            status=DecisionStatus.CONTINUE if len(calls) == 1 else DecisionStatus.SATISFIED,
        )

    monkeypatch.setattr("agent_runtime.strategies.react.graph.run_decision", fake_decision)
    monkeypatch.setattr("agent_runtime.strategies.react.graph.run_executor", lambda current: current)

    result = strategy.run(state, thread_id="test")

    assert calls == ["planner", "planner"]
    assert result.node_states.planner.plan is not None
    assert result.node_states.planner.plan_count == 2


def test_react_stops_at_max_turns(react_agent_state_factory, plan_factory, plan_step_factory, monkeypatch) -> None:
    strategy = ReactStrategy()
    state = react_agent_state_factory(max_turns=1)
    calls: list[str] = []
    action = plan_factory(steps=[plan_step_factory(id="E1", plan="lookup", tool="lookup", args={})])

    def fake_decision(state, *, stage, callsite, prompt_builder, message_history=None):
        calls.append(stage)
        return DecisionResult(plan=action, status=DecisionStatus.CONTINUE)

    monkeypatch.setattr("agent_runtime.strategies.react.graph.run_decision", fake_decision)
    monkeypatch.setattr("agent_runtime.strategies.react.graph.run_executor", lambda current: current)

    strategy.run(state, thread_id="test")

    assert calls == ["planner"]


def test_react_stops_when_latest_plan_is_entirely_duplicate_rejected(react_agent_state_factory, plan_factory, plan_step_factory) -> None:
    strategy = ReactStrategy()
    state = react_agent_state_factory()
    step = plan_step_factory(id="E1", plan="lookup", tool="lookup", args={})
    state.node_states.planner.plan = plan_factory(steps=[step])
    state.direct_tool_results.append(
        ToolResult(
            plan_step_id=step.db_id,
            tool_name="lookup",
            result={
                "error": "Tool request for 'lookup' was already made in this roundtrip.",
                "status": "duplicate",
            },
        )
    )

    route = strategy._route_after_execute(state)

    assert route == "__end__"
    assert state.node_states.planner.decision_status == DecisionStatus.BLOCKED
    assert state.result.execution_status.value == "no_new_work"


def test_react_records_explicit_blocked_decision(react_agent_state_factory, plan_factory, monkeypatch) -> None:
    strategy = ReactStrategy()
    state = react_agent_state_factory()

    monkeypatch.setattr(
        "agent_runtime.strategies.react.graph.run_decision",
        lambda *args, **kwargs: DecisionResult(
            plan=plan_factory(steps=[]),
            status=DecisionStatus.BLOCKED,
        ),
    )

    result = strategy.run(state, thread_id="test")

    assert result.node_states.planner.decision_status == DecisionStatus.BLOCKED
    assert result.result.answer_status.value == "failed"


def test_direct_return_reasons_once(react_agent_state_factory, plan_factory, plan_step_factory, monkeypatch) -> None:
    strategy = DirectReturnStrategy()
    state = react_agent_state_factory()
    calls: list[str] = []
    action = plan_factory(steps=[plan_step_factory(id="E1", plan="lookup", tool="lookup", args={})])

    def fake_decision(state, *, stage, callsite, prompt_builder):
        calls.append(callsite)
        return DecisionResult(plan=action, status=DecisionStatus.CONTINUE)

    monkeypatch.setattr(
        "agent_runtime.strategies.single_pass.graph.run_decision",
        fake_decision,
    )
    monkeypatch.setattr(
        "agent_runtime.strategies.single_pass.graph.run_executor",
        lambda current: current,
    )

    result = strategy.run(state, thread_id="test")

    assert calls == ["direct_return.reasoner"]
    assert result.node_states.planner.plan_count == 1


@pytest.mark.parametrize("payload, expected", [
    ({"ok": True}, "success"),
    ({"error": "Provider failed"}, "failed"),
    ({"status": "failed"}, "failed"),
    ({"status": "duplicate"}, "failed"),
])
def test_direct_return_classifies_in_memory_results(react_agent_state_factory, plan_factory, plan_step_factory, payload, expected) -> None:
    state = react_agent_state_factory()
    step = plan_step_factory(id="E1", tool="lookup", plan="Look up data")
    state.node_states.planner.plan = plan_factory(steps=[step])
    state.direct_tool_results = [ToolResult(plan_step_id=step.db_id, result=payload)]

    DirectReturnStrategy._route_after_execute(state)

    assert state.result.execution_status.value == expected


@pytest.mark.parametrize("kind", ["builtin", "user_agent"])
@pytest.mark.parametrize("conversation_id", [None, "conversation-123"])
def test_agent_registry_runs_profile_strategy_with_conversation_thread(agent_state_factory, agent_profile_factory, monkeypatch, kind, conversation_id) -> None:
    profile = agent_profile_factory(
        name="custom_agent", scope="main_agent", kind=AgentKind(kind),
        execution_strategy=AgentExecutionStrategy.REACT,
        planner_instruction="Use this exact instruction.",
    )
    state = agent_state_factory(profile=profile, llm=object())
    state.execution_context.conversation_id = conversation_id
    seen = []

    def fake_run(self, current, *, thread_id):
        seen.append((current, thread_id))
        return current

    monkeypatch.setattr(ReactStrategy, "run", fake_run)
    result = AgentRegistry().get(profile)(state)

    assert result is state
    assert seen == [(state, conversation_id or "")]
    assert state.agent_profile is profile
