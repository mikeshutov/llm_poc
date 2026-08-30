from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from uuid import uuid4

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
from request_orchestrator.models.evidence import EvidenceView, ToolResult
from request_orchestrator.models.main_state import MainState


def test_main_state_gathers_typed_child_results_without_rebasing_ids() -> None:
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
    profile_step_id = uuid4()
    profile_result = ToolResult(
        plan_step_id=profile_step_id,
        tool_name="get_current_weather",
        evidence=[
            EvidenceView(
                item_id="Toronto",
                title="Weather Result",
                summary="21.2 C in Toronto.",
                source="get_current_weather",
                entity_type="weather",
            )
        ],
    )
    profile_state.gather_tool_results = lambda: [profile_result]
    profile_state.result = profile_state.result.copy(relevant_evidence_ids=[uuid4()])

    main_agent_state = main_state.agent_states["main_agent"]
    main_step_id = uuid4()
    main_result = ToolResult(
        plan_step_id=main_step_id,
        tool_name="generic_web_search",
        evidence=[
            EvidenceView(
                item_id="ramen-1",
                title="Ramen Result",
                summary="Popular ramen shop.",
                source="generic_web_search",
                entity_type="web_search_results",
            )
        ],
    )
    main_agent_state.gather_tool_results = lambda: [main_result]
    main_agent_state.result = main_agent_state.result.copy(relevant_evidence_ids=[uuid4()])

    tool_results = main_state.gather_tool_results()
    relevant_evidence_ids = main_state.gather_relevant_evidence_ids()

    assert [tool_result.plan_step_id for tool_result in tool_results] == [profile_step_id, main_step_id]
    assert len(relevant_evidence_ids) == 2
