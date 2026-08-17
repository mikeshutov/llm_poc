from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

if "yfinance" not in sys.modules:
    sys.modules["yfinance"] = ModuleType("yfinance")

if "pycountry" not in sys.modules:
    pycountry_module = ModuleType("pycountry")
    pycountry_module.countries = SimpleNamespace(
        lookup=lambda value: SimpleNamespace(alpha_2=str(value).upper())
    )
    sys.modules["pycountry"] = pycountry_module

from conversation.models.conversation_models import ConversationContext
from personalization.profile.models import UserProfile
from request_orchestrator.agents.main_agent.profile import MAIN_AGENT_PROFILE
from request_orchestrator.agents.profile_management.profile import build_profile_management_profile
from request_orchestrator.agents.registry import agent_registry
from request_orchestrator.agent_runner.models.agent_profile import AgentProfile
from request_orchestrator.models.agent_execution_context import AgentExecutionContext
from request_orchestrator.models.orchestrator_graph_state import OrchestratorGraphState
from request_orchestrator.models.agent_result import AgentResult
from request_orchestrator.models.evidence import ToolResult
from request_orchestrator.models.main_state import MainState
from request_orchestrator.nodes.agent_execution_nodes import run_single_agent_node
from request_orchestrator.nodes.main_state_nodes import apply_agent_updates_node
from request_orchestrator.orchestrator import (
    OrchestratorGraph,
    _ORCHESTRATOR,
)


def test_run_single_agent_returns_agent_update() -> None:
    user_profile = UserProfile()
    main_state = MainState.new(
        task="Run child agent.",
        execution_context=AgentExecutionContext.new(
            conversation_context=ConversationContext(),
            user_profile=user_profile,
        ),
        llm=object(),
        agent_profiles=[
            build_profile_management_profile(user_profile),
            MAIN_AGENT_PROFILE,
        ],
    )

    main_agent_state = main_state.agent_states["main_agent"]
    main_agent_state.inputs.task = "Find evidence"

    def fake_runner(agent_state):
        updated_state = agent_state
        updated_state.result = AgentResult(
            tool_results=[
                ToolResult(
                    step_id="P1E1",
                    tool_name="generic_web_search",
                    result={"items": ["result"]},
                )
            ],
            relevant_evidence_ids=["P1E1R1"],
        )
        return updated_state

    previous_registry_get = agent_registry.get
    agent_registry.get = lambda agent_name: fake_runner if agent_name == "main_agent" else previous_registry_get(agent_name)
    try:
        update_payload = run_single_agent_node(main_state.agent_states["main_agent"])
    finally:
        agent_registry.get = previous_registry_get

    updates = update_payload["completed_agents"]
    assert list(updates) == ["main_agent"]
    updated_agent_state = updates["main_agent"]
    assert len(updated_agent_state.result.tool_results) == 1
    assert updated_agent_state.result.tool_results[0].step_id == "P1E1"
    assert updated_agent_state.result.relevant_evidence_ids == ["P1E1R1"]


def test_apply_agent_updates_merges_returned_agent_state_into_main_state() -> None:
    user_profile = UserProfile()
    main_state = MainState.new(
        task="Run child agent.",
        execution_context=AgentExecutionContext.new(
            conversation_context=ConversationContext(),
            user_profile=user_profile,
        ),
        llm=object(),
        agent_profiles=[
            build_profile_management_profile(user_profile),
            MAIN_AGENT_PROFILE,
        ],
    )

    updated_agent_state = main_state.agent_states["main_agent"]
    updated_agent_state.result = AgentResult(
        tool_results=[
            ToolResult(
                step_id="P1E1",
                tool_name="generic_web_search",
                result={"items": ["result"]},
            )
        ],
        relevant_evidence_ids=["P1E1R1"],
    )

    payload = apply_agent_updates_node(
        OrchestratorGraphState(
            main_state=main_state,
            completed_agents={
                "main_agent": updated_agent_state,
            },
        )
    )

    merged_main_state = payload["main_state"]
    assert len(merged_main_state.agent_states["main_agent"].result.tool_results) == 1
    assert merged_main_state.agent_states["main_agent"].result.tool_results[0].step_id == "P1E1"
    assert merged_main_state.agent_states["main_agent"].result.relevant_evidence_ids == ["P1E1R1"]


def test_orchestrator_graph_is_compiled() -> None:
    assert isinstance(_ORCHESTRATOR, OrchestratorGraph)


def test_run_single_agent_rejects_missing_agent_modules() -> None:
    user_profile = UserProfile()
    unknown_profile = AgentProfile(name="unknown_agent", scope="main_agent")
    main_state = MainState.new(
        task="Run child agent.",
        execution_context=AgentExecutionContext.new(
            conversation_context=ConversationContext(),
            user_profile=user_profile,
        ),
        llm=object(),
        agent_profiles=[unknown_profile],
    )
    unknown_agent_state = main_state.agent_states["unknown_agent"]

    try:
        run_single_agent_node(unknown_agent_state)
    except ModuleNotFoundError as exc:
        assert "unknown_agent" in str(exc)
    else:
        raise AssertionError("Expected missing agent module lookup to fail")
