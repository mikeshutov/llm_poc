from __future__ import annotations

from langsmith import traceable

from personalization.profile.service import load_user_profile_attributes
from request_orchestrator.models.agent_state import AgentState


@traceable(name="Load User Profile Node")
def load_user_profile(agent_state: AgentState) -> AgentState:
    load_user_profile_attributes(
        agent_state.user_profile,
        requested_attribute_types=agent_state.request_analysis.requested_user_attribute_types,
    )
    return agent_state
