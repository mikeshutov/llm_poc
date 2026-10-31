from __future__ import annotations

from agent_runtime.models.agent_profile import AgentProfile
from agent_runtime.models.agent_state import AgentState
from agent_runtime.models.agent_execution_context import AgentExecutionContext
from agent_runtime.models.plan import Plan, PlanStep
from capabilities.discovery.models import AvailableAgent, AvailableTool, CapabilityDiscoveryResult
from capabilities.discovery.discover_capabilities_tool import (
    CAPABILITY_DISCOVERY_TOOL,
    bind_capability_discovery_agent_state,
    bind_capability_discovery_state,
    reset_capability_discovery_agent_state,
    reset_capability_discovery_state,
)
from capabilities.adapters.profile import (
    USER_MANAGEMENT_TOOL,
    bind_user_management_agent_state,
    bind_user_management_state,
    reset_user_management_agent_state,
    reset_user_management_state,
)
from agents.profile_management.profile import build_profile_management_profile
from agent_runtime.models.main_state import MainState
from agents.main.agent import MainReasonerStrategy
from agents.main import agent as main_reasoner
from agent_runtime.strategies.react.decision import DecisionResult, DecisionStatus


def test_main_reasoner_loops_until_final_structured_answer(monkeypatch) -> None:
    state = AgentState.new(
        agent_profile=AgentProfile(name="request_orchestrator", scope="main_agent", max_turns=3),
        task="Find the answer",
        execution_context=AgentExecutionContext.new(),
        llm=object(),
    )
    calls = []
    plan = Plan(steps=[PlanStep(id="E1", plan="Look it up", tool="lookup", args={})])

    def fake_reasoner(*args, **kwargs):
        calls.append("reasoner")
        if len(calls) == 1:
            return DecisionResult(plan=plan, status=DecisionStatus.CONTINUE)
        return DecisionResult(
            plan=Plan(steps=[]),
            status=DecisionStatus.SATISFIED,
            raw_content='{"result":[{"content":"final answer","evidence_ids":[]}],"next_question":"","roundtrip_summary":"done"}',
        )

    monkeypatch.setattr(main_reasoner, "run_decision", fake_reasoner)
    monkeypatch.setattr(main_reasoner, "run_executor", lambda current: current)

    result = MainReasonerStrategy().run(state, thread_id="test")

    assert calls == ["reasoner", "reasoner"]
    assert result.reasoner_result is not None
    assert result.reasoner_result.result[0].content == "final answer"
    assert result.node_states.planner.plan_count == 2


def test_main_reasoner_does_not_receive_planner_only_tool_constraints() -> None:
    state = AgentState.new(
        agent_profile=AgentProfile(name="request_orchestrator", scope="main_agent"),
        task="Research this topic",
        execution_context=AgentExecutionContext.new(),
        llm=object(),
    )

    prompt, _ = main_reasoner.build_reasoner_prompt(state)
    rendered = prompt.build()

    assert "Do not make one planned tool step depend on another step's output" not in rendered
    assert "Use at most ONE web search tool call in the entire plan" not in rendered
    assert "Tool calls may be planned across multiple reasoner turns" in rendered


def test_runtime_discovery_tool_is_static_and_updates_active_capabilities(monkeypatch) -> None:
    main_state = MainState.new(
        task="Find a meal",
        execution_context=AgentExecutionContext.new(),
        llm=object(),
        agent_profiles=[],
    )
    agent_state = AgentState.new(
        agent_profile=AgentProfile(name="request_orchestrator", scope="main_agent"),
        execution_context=main_state.execution_context,
        llm=object(),
    )
    expected = CapabilityDiscoveryResult(
        tools=[AvailableTool(name="food_search", description="Find food")],
        agents=[AvailableAgent(agent="meal_agent", description="Find meals")],
    )
    monkeypatch.setattr("capabilities.discovery.discover_capabilities_tool.capability_discovery", lambda *args, **kwargs: expected)
    state_token = bind_capability_discovery_state(main_state)
    agent_token = bind_capability_discovery_agent_state(agent_state)
    try:
        output = CAPABILITY_DISCOVERY_TOOL.invoke({"prompt": "a recipe search tool"})
    finally:
        reset_capability_discovery_agent_state(agent_token)
        reset_capability_discovery_state(state_token)

    assert CAPABILITY_DISCOVERY_TOOL.name == "discover_capabilities"
    assert output["tools"][0]["name"] == "food_search"
    assert main_state.discovered_capabilities == expected
    assert agent_state.inputs.discovered_capabilities == expected


def test_user_management_static_tool_runs_profile_agent(monkeypatch) -> None:
    main_state = MainState.new(
        task="Remember my preference",
        execution_context=AgentExecutionContext.new(),
        llm=object(),
        agent_profiles=[build_profile_management_profile()],
    )
    agent_state = AgentState.new(
        agent_profile=AgentProfile(name="request_orchestrator", scope="main_agent"),
        execution_context=main_state.execution_context,
        llm=object(),
    )
    captured: list[str] = []

    def fake_runner(profile_state):
        captured.append(profile_state.inputs.task)
        return profile_state

    monkeypatch.setattr("agents.registry.agent_registry.get", lambda profile: fake_runner)
    state_token = bind_user_management_state(main_state)
    agent_token = bind_user_management_agent_state(agent_state)
    try:
        output = USER_MANAGEMENT_TOOL.invoke({"task": "Remember that I like spicy food."})
    finally:
        reset_user_management_agent_state(agent_token)
        reset_user_management_state(state_token)

    assert captured == ["Remember that I like spicy food."]
    assert output["agent"] == "profile_management"
    assert output["status"] == "pending"
