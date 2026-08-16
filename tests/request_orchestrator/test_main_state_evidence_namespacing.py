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
from request_orchestrator.models.evidence import EvidenceView, HydratedEvidence, ToolResult
from request_orchestrator.models.main_state import MainState
from request_orchestrator.models.plan_step_ids import namespace_evidence_id, namespace_step_id


def test_main_state_gathers_child_results_without_rebasing_ids() -> None:
    user_profile = UserProfile()
    main_state = MainState.new(
        task="Combine child evidence.",
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

    profile_state = main_state.agent_states["profile_management"]
    profile_state.result = AgentResult(
        tool_results=[
            ToolResult(
                step_id=namespace_step_id("profile_management", "P1E1"),
                tool_name="get_current_weather",
                evidence_views=[
                    EvidenceView(
                        item_id="Toronto",
                        title="Weather Result",
                        summary="21.2 C in Toronto.",
                    )
                ],
                hydrated_evidence=[
                    HydratedEvidence(
                        item_id="Toronto",
                        title="Weather Result",
                        summary="21.2 C in Toronto.",
                        source="get_current_weather",
                        entity_type="weather",
                    )
                ],
            )
        ],
        relevant_evidence_ids=[namespace_evidence_id("profile_management", "P1E1R1")],
    )

    main_agent_state = main_state.agent_states["main_agent"]
    main_agent_state.result = AgentResult(
        tool_results=[
            ToolResult(
                step_id=namespace_step_id("main_agent", "P1E1"),
                tool_name="generic_web_search",
                evidence_views=[
                    EvidenceView(
                        item_id="ramen-1",
                        title="Ramen Result",
                        summary="Popular ramen shop.",
                    )
                ],
                hydrated_evidence=[
                    HydratedEvidence(
                        item_id="ramen-1",
                        title="Ramen Result",
                        summary="Popular ramen shop.",
                        source="generic_web_search",
                        entity_type="web_search_results",
                    )
                ],
            )
        ],
        relevant_evidence_ids=[namespace_evidence_id("main_agent", "P1E1R1")],
    )

    tool_results = main_state.gather_tool_results()
    relevant_evidence_ids = main_state.gather_relevant_evidence_ids()

    assert [tool_result.step_id for tool_result in tool_results] == [
        "profile_management:P1E1",
        "main_agent:P1E1",
    ]
    assert relevant_evidence_ids == [
        "profile_management:P1E1R1",
        "main_agent:P1E1R1",
    ]
