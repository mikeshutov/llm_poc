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
from request_orchestrator.models.agent_execution_context import AgentExecutionContext
from request_orchestrator.models.agent_result import AgentResult
from request_orchestrator.models.evidence import ToolResult
from request_orchestrator.models.main_state import MainState
from request_orchestrator.orchestrator import _build_run_agent_node


def test_run_agent_node_persists_returned_agent_state_into_main_state() -> None:
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

    run_agent_node = _build_run_agent_node("main_agent", fake_runner)
    updated_main_state = run_agent_node(main_state)

    updated_agent_state = updated_main_state.agent_states["main_agent"]
    assert len(updated_agent_state.result.tool_results) == 1
    assert updated_agent_state.result.tool_results[0].step_id == "P1E1"
    assert updated_agent_state.result.relevant_evidence_ids == ["P1E1R1"]
