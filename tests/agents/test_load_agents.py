import pytest

from agents.discovery import load_agents as load_agents_module
from agents.document_writing.profile import DOCUMENT_WRITING_PROFILE
from agents.discovery.load_agents import load_agents
from agents.research.profile import RESEARCH_PROFILE
from agents.delegate.profile import DELEGATE_PROFILE
from agents.coding.profile import CODING_PROFILE
from agents.profile_management.profile import PROFILE_MANAGEMENT_PROFILE
from agent_runtime.models.agent_execution_context import AgentExecutionContext
from agent_runtime.models.agent_profile import AgentExecutionStrategy, AgentKind
from capabilities.registry.constants import (
    TOOL_CATEGORY_MATH,
)
from agent_runtime.models.agent_result import AnswerStatus, ExecutionStatus
from agent_runtime.execution.tool_executor import run_executor
from agent_runtime.planning.tool_call_parser import parse_tool_calls_to_plan_steps
from agent_runtime.planning.tool_wrapper import build_tools
from agent_runtime.strategies.react.decision import build_decision_prompt
from agents.models.agent import AgentModelConfig, AgentType
from agents.registry import agent_registry
from capabilities.discovery.models import AvailableAgent, CapabilityDiscoveryResult
from capabilities.registry.global_registry import GLOBAL_TOOL_REGISTRY, REGISTERED_AGENTS
from personalization.profile.models import UserProfile


def test_load_agents_adds_system_agents_before_user_agent_discovery(main_state_factory, monkeypatch) -> None:
    monkeypatch.setattr(load_agents_module, "embed_text", lambda _: [1.0])

    class FakeRepository:
        def list_for_user(self, user_id):
            return []

    monkeypatch.setattr(load_agents_module, "get_agent_repo", lambda: FakeRepository())

    state = main_state_factory(
        task="research a topic and write the findings",
        execution_context=AgentExecutionContext.new(),
        llm=object(),
        agent_profiles=[PROFILE_MANAGEMENT_PROFILE],
    )

    load_agents(state)

    assert state.agent_profiles == [
        PROFILE_MANAGEMENT_PROFILE,
        DOCUMENT_WRITING_PROFILE,
        RESEARCH_PROFILE,
        CODING_PROFILE,
        DELEGATE_PROFILE,
    ]
    assert set(state.agent_states) == {
        "profile_management",
        "document_writing",
        "research",
        "coding",
        "delegate",
    }


@pytest.mark.parametrize("strategy", list(AgentExecutionStrategy))
def test_db_agents_are_exposed_and_executed_for_their_owner(
    main_state_factory,
    plan_factory,
    agent_profile_factory,
    monkeypatch,
    strategy,
    db_agent_factory,
    agent_state_factory,
) -> None:
    rows = {
        owner: db_agent_factory(
            agent_type=AgentType.USER, user_id=owner, name="db_helper",
            description=f"{owner}'s helper loaded from the database",
            execution_strategy=strategy, planner_instruction=f"Follow {owner}'s instructions.",
            allowed_categories=[TOOL_CATEGORY_MATH], max_turns=4,
            model_configs=[AgentModelConfig(stage="planner", provider="openai", model="gpt-4o-mini")],
        )
        for owner in ("alice", "bob")
    }

    class Repository:
        def list_for_user(self, user_id):
            return [rows[user_id]]

        def list_relevant_for_user(self, user_id, *, query_embedding):
            assert query_embedding == [1.0]
            return [rows[user_id]]

    monkeypatch.setattr(load_agents_module, "embed_text", lambda query: [1.0])
    monkeypatch.setattr(load_agents_module, "get_agent_repo", Repository)
    monkeypatch.setattr(
        "agent_runtime.execution.tool_executor.capability_discovery",
        lambda state, **kwargs: CapabilityDiscoveryResult(),
    )
    parents = []
    for owner in rows:
        context = AgentExecutionContext.new(
            user_profile=UserProfile(user_id=owner), conversation_id=f"conversation-{owner}",
        )
        main = main_state_factory(
            task="Delegate a calculation",
            execution_context=context,
            llm=object(),
            agent_profiles=[],
        )
        load_agents(main)
        child = main.agent_states["db_helper"]
        assert child.agent_profile.kind == AgentKind.USER_AGENT
        assert child.agent_profile.execution_strategy == strategy
        assert child.agent_profile.max_turns == 4
        assert child.agent_profile.allowed_categories == {TOOL_CATEGORY_MATH}
        assert child.agent_profile.stage_model_selections["planner"].model == "gpt-4o-mini"
        parent = agent_state_factory(
            profile=agent_profile_factory(name="coordinator", scope="main_agent"),
            execution_context=context,
        )
        parent.available_agent_states = main.agent_states
        parent.inputs.discovered_capabilities = CapabilityDiscoveryResult(agents=[AvailableAgent(agent="db_helper")])
        parents.append(parent)

    seen = []

    def run_strategy(self, state, *, thread_id):
        owner = state.execution_context.user_profile.user_id
        seen.append((owner, state.agent_profile.planner_instruction, thread_id))
        assert state.inputs.task == "Calculate the requested value."
        assert state.inputs.evidence_ids == ["ev-1"]
        state.result = state.result.copy(answer_status=AnswerStatus.SUCCESS, execution_status=ExecutionStatus.SUCCESS)
        if strategy == AgentExecutionStrategy.PLANNER_EXECUTOR_EVALUATOR:
            state.node_states.evaluator.evaluation_status = "SATISFIED"
        return state

    monkeypatch.setattr(type(agent_registry._strategies[strategy]), "run", run_strategy)
    for parent in parents:
        owner = parent.execution_context.user_profile.user_id
        prompt, tools = build_decision_prompt(parent, instruction="Select a tool", rules="")
        exposed = next(tool for tool in tools if tool.name == "db_helper")
        assert exposed is GLOBAL_TOOL_REGISTRY.get("db_helper", user_id=owner)
        assert exposed.description == rows[owner].description
        assert "db_helper" in {tool.name for tool in prompt.available_tools}
        provider_tool = next(tool for tool in build_tools(tools) if tool.name == "db_helper")
        args = {
            "parameters": {"task": "Calculate the requested value.", "evidence_ids": ["ev-1"]},
            "hypothesis": "A result is calculated.",
        }
        provider_tool.args_schema.model_validate(args)
        parent.node_states.planner.plan = plan_factory(steps=parse_tool_calls_to_plan_steps([
            {"id": "E1", "name": "db_helper", "args": args},
        ], tools))
        run_executor(parent)
        result = parent.direct_tool_results[0]
        assert result.tool_name == "db_helper"
        assert result.result["status"] == "success"
        assert not result.result.get("error")

    assert seen == [
        (owner, rows[owner].planner_instruction, f"conversation-{owner}") for owner in rows
    ]
    assert "db_helper" not in {profile.name for profile in REGISTERED_AGENTS}
