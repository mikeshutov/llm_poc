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
from agent_runtime.execution.message_history import append_execution_messages
from agents.main import agent as main_reasoner
from agent_runtime.models.decision import DecisionStatus
from agent_runtime.strategies.react.decision import DecisionResult
from agent_runtime.models.agent_prompt import EvidenceStep
from agent_runtime.models.synthesized_result import SynthesisResultBlock
from retrieval.evidence import ToolResult
from retrieval.evidence import EvidenceUrl, EvidenceView
from agents.main.request import _normalize_generated_file_links
from agents.main.request import TOP_LEVEL_PROFILE


def test_normalize_generated_file_links_uses_file_evidence_url() -> None:
    evidence = EvidenceView(
        title="report.pdf",
        entity_type="file",
        urls=[EvidenceUrl(url="/app/static/files/anonymous/report.pdf")],
    )
    blocks = _normalize_generated_file_links(
        [SynthesisResultBlock(content="[Download](sandbox:/mnt/data/report.pdf)")],
        [EvidenceStep(type="file", evidence=[evidence])],
    )

    assert blocks[0].content == "[Download](/app/static/files/anonymous/report.pdf)"


def test_normalize_generated_file_links_replaces_app_static_file_links() -> None:
    evidence = EvidenceView(
        title="chart.png",
        entity_type="file",
        urls=[EvidenceUrl(url="/app/static/files/user-123/chart.png")],
    )
    blocks = _normalize_generated_file_links(
        [SynthesisResultBlock(content="[Download](sandbox:/app/static/files/user-123/chart.png)")],
        [EvidenceStep(type="file", evidence=[evidence])],
    )

    assert blocks[0].content == "[Download](/app/static/files/user-123/chart.png)"


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


def test_main_reasoner_does_not_force_a_tool_on_the_first_turn(monkeypatch) -> None:
    state = AgentState.new(
        agent_profile=AgentProfile(name="request_orchestrator", scope="main_agent", max_turns=1),
        task="Who wrote The Three-Body Problem?",
        execution_context=AgentExecutionContext.new(),
        llm=object(),
    )
    captured = {}

    def fake_decision(*args, **kwargs):
        captured.update(kwargs)
        return DecisionResult(plan=Plan(steps=[]), status=DecisionStatus.SATISFIED, raw_content="{}")

    monkeypatch.setattr(main_reasoner, "run_decision", fake_decision)
    monkeypatch.setattr(main_reasoner, "run_executor", lambda current: current)

    MainReasonerStrategy().run(state, thread_id="test")

    assert captured.get("tool_choice") is None


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
    assert "Ground factual claims, citations, URLs, and generated document content" in rendered


def test_main_reasoner_represents_agents_as_tool_shaped_capabilities() -> None:
    state = AgentState.new(
        agent_profile=AgentProfile(name="request_orchestrator", scope="main_agent"),
        task="Research this topic",
        execution_context=AgentExecutionContext.new(),
        llm=object(),
    )
    delegated = AgentState.new(
        agent_profile=AgentProfile(
            name="research",
            scope="main_agent",
            description="Research a topic using multiple sources.",
        ),
        execution_context=state.execution_context,
        llm=object(),
    )
    state.available_agent_states = {"research": delegated}
    state.inputs.discovered_capabilities = CapabilityDiscoveryResult(
        agents=[AvailableAgent(agent="research", description=delegated.agent_profile.description)]
    )

    prompt, tools = main_reasoner.build_reasoner_prompt(state)

    assert all(isinstance(item, AvailableTool) for item in prompt.available_tools)
    capabilities = prompt.sections_raw["available_tools"]
    assert all(isinstance(item, dict) for item in capabilities)
    research_capability = next(item for item in capabilities if item["name"] == "research")
    assert research_capability["category"] == "agents"
    assert "agent" not in research_capability
    assert "research" in {tool.name for tool in tools}


def test_top_level_reasoner_has_direct_math_and_prefers_it_over_coding() -> None:
    assert "math_calculate" in TOP_LEVEL_PROFILE.tool_names
    assert "math_calculate" in TOP_LEVEL_PROFILE.planner_instruction
    assert "coding agent" in TOP_LEVEL_PROFILE.planner_instruction


def test_top_level_reasoner_does_not_receive_subagent_decision_contract() -> None:
    state = AgentState.new(
        agent_profile=TOP_LEVEL_PROFILE,
        task="Find the answer",
        execution_context=AgentExecutionContext.new(),
        llm=object(),
    )

    prompt, _ = main_reasoner.build_reasoner_prompt(state)

    assert "OUTPUT CONTRACT" not in prompt.build()


def test_top_level_reasoner_uses_profile_management_facade() -> None:
    state = AgentState.new(
        agent_profile=TOP_LEVEL_PROFILE,
        task="Remember that I prefer concise technical explanations.",
        execution_context=AgentExecutionContext.new(),
        llm=object(),
    )
    state.inputs.discovered_capabilities = CapabilityDiscoveryResult(
        tools=[AvailableTool(name="user_update_tone", description="Update tone")]
    )

    _, tools = main_reasoner.build_reasoner_prompt(state)

    assert "user_update_tone" not in {tool.name for tool in tools}
    assert "user_management" in {tool.name for tool in tools}


def test_top_level_reasoner_exposes_memory_detail_with_search_guidance() -> None:
    state = AgentState.new(
        agent_profile=TOP_LEVEL_PROFILE,
        task="What did we previously discuss about agent delegation?",
        execution_context=AgentExecutionContext.new(),
        llm=object(),
    )
    state.inputs.discovered_capabilities = CapabilityDiscoveryResult(
        tools=[
            AvailableTool(name="memory_search", description="Search memories"),
            AvailableTool(name="memory_detail", description="Read memory detail"),
        ]
    )

    _, tools = main_reasoner.build_reasoner_prompt(state)

    assert "memory_search" in {tool.name for tool in tools}
    assert "memory_detail" in {tool.name for tool in tools}


def test_main_reasoner_prompt_prefers_research_agent_for_multi_source_comparisons() -> None:
    state = AgentState.new(
        agent_profile=TOP_LEVEL_PROFILE,
        task="Compare LangGraph and CrewAI for multi-agent orchestration.",
        execution_context=AgentExecutionContext.new(),
        llm=object(),
    )

    prompt, _ = main_reasoner.build_reasoner_prompt(state)

    assert "prefer the research agent" in prompt.rules
    assert "do not duplicate" in prompt.rules


def test_main_reasoner_routes_structured_data_files_to_coding() -> None:
    state = AgentState.new(
        agent_profile=TOP_LEVEL_PROFILE,
        task="Create a CSV with ten sample sales records.",
        execution_context=AgentExecutionContext.new(),
        llm=object(),
    )

    prompt, _ = main_reasoner.build_reasoner_prompt(state)

    assert "CSV or other structured data-file generation" in prompt.rules
    assert "Do not route data files through document_writing" in prompt.rules


def test_main_reasoner_routes_source_code_artifacts_to_coding() -> None:
    state = AgentState.new(
        agent_profile=TOP_LEVEL_PROFILE,
        task="Create a Python script that calculates compound interest.",
        execution_context=AgentExecutionContext.new(),
        llm=object(),
    )

    prompt, _ = main_reasoner.build_reasoner_prompt(state)

    assert "Python scripts or other source-code artifacts" in prompt.rules
    assert "Do not wrap source code in Markdown" in prompt.rules


def test_main_reasoner_allows_generated_data_for_explicit_sample_charts() -> None:
    state = AgentState.new(
        agent_profile=TOP_LEVEL_PROFILE,
        task="Generate a scatterplot of sample data with a regression line.",
        execution_context=AgentExecutionContext.new(),
        llm=object(),
    )

    prompt, _ = main_reasoner.build_reasoner_prompt(state)

    assert "underspecified" in prompt.rules
    assert "safe, conventional, reversible default" in prompt.rules
    assert "generate a small clearly labeled dataset" in prompt.rules


def test_main_reasoner_uses_safe_default_for_unspecified_multiplication_table() -> None:
    state = AgentState.new(
        agent_profile=TOP_LEVEL_PROFILE,
        task="Generate a multiplication table and save it as CSV.",
        execution_context=AgentExecutionContext.new(),
        llm=object(),
    )

    prompt, _ = main_reasoner.build_reasoner_prompt(state)

    assert "use 1 through 10" in prompt.rules
    assert "Ask a clarifying question only when" in prompt.rules


def test_main_reasoner_appends_subagent_result_to_next_llm_message_history() -> None:
    state = AgentState.new(
        agent_profile=AgentProfile(name="request_orchestrator", scope="main_agent"),
        execution_context=AgentExecutionContext.new(),
        llm=object(),
    )
    state.node_states.planner.plan = Plan(steps=[PlanStep(
        id="E1",
        plan="Create the document.",
        agent="document_writing",
        args={"task": "Create the document."},
    )])
    state.direct_tool_results.append(ToolResult(
        plan_step_id=state.node_states.planner.plan.steps[0].db_id,
        tool_name="document_writing",
        result={
            "agent": "document_writing",
            "status": "success",
            "summary": "File created.",
            "evidence": [],
        },
    ))

    append_execution_messages(state)

    assert len(state.reasoner_message_history) == 2
    assert state.reasoner_message_history[0].tool_calls[0]["name"] == "document_writing"
    assert state.reasoner_message_history[0].tool_calls[0]["args"] == {
        "task": "Create the document.",
        "evidence_ids": [],
    }
    assert "File created." in state.reasoner_message_history[1].content


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
    assert output.agent == "profile_management"
    assert output.status.value == "pending"
